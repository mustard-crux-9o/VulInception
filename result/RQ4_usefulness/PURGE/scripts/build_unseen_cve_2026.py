#!/usr/bin/env python3
import argparse
import ast
import base64
import difflib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
from packaging.version import InvalidVersion, Version


COMMIT_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/commit/([0-9a-fA-F]{7,40})")
CVE_RE = re.compile(r"CVE-2026-\d+", re.IGNORECASE)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
BANDIT_CWES = {
    "CWE-020",
    "CWE-022",
    "CWE-078",
    "CWE-079",
    "CWE-089",
    "CWE-094",
    "CWE-259",
    "CWE-295",
    "CWE-327",
    "CWE-377",
    "CWE-502",
    "CWE-611",
    "CWE-703",
    "CWE-798",
}


def request_json(url, headers=None, params=None, sleep=0.2):
    time.sleep(sleep)
    response = requests.get(url, headers=headers or {}, params=params, timeout=60)
    if response.status_code == 403 and "rate limit" in response.text.lower():
        reset = int(response.headers.get("x-ratelimit-reset", "0") or 0)
        wait = max(5, min(120, reset - int(time.time()) + 2))
        time.sleep(wait)
        response = requests.get(url, headers=headers or {}, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def request_text(url, headers=None, sleep=0.2):
    time.sleep(sleep)
    response = requests.get(url, headers=headers or {}, timeout=60)
    response.raise_for_status()
    return response.text


def request_bytes(url, headers=None, sleep=0.2):
    time.sleep(sleep)
    response = requests.get(url, headers=headers or {}, timeout=120)
    response.raise_for_status()
    return response.content


def date_chunks(start, end, days=90):
    current = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while current <= stop:
        chunk_end = min(stop, current + timedelta(days=days - 1))
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + timedelta(days=1)


def nvd_range(start, end):
    return f"{start}T00:00:00.000", f"{end}T23:59:59.999"


def iter_nvd_cves(start, end, limit_nvd):
    yielded = 0
    for begin, finish in date_chunks(start, end):
        index = 0
        while True:
            pub_start, pub_end = nvd_range(begin, finish)
            data = request_json(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                params={
                    "pubStartDate": pub_start,
                    "pubEndDate": pub_end,
                    "resultsPerPage": 2000,
                    "startIndex": index,
                },
                sleep=0.6,
            )
            rows = data.get("vulnerabilities") or []
            for row in rows:
                yield row.get("cve") or {}
                yielded += 1
                if limit_nvd and yielded >= limit_nvd:
                    return
            total = int(data.get("totalResults") or 0)
            index += len(rows)
            if not rows or index >= total:
                break


def cwe_ids(cve):
    values = []
    for weakness in cve.get("weaknesses") or []:
        for desc in weakness.get("description") or []:
            value = desc.get("value")
            if value and value.startswith("CWE-"):
                values.append(value)
    return sorted(set(values))


def github_headers(token):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_search_headers(token):
    headers = github_headers(token)
    headers["Accept"] = "application/vnd.github.cloak-preview+json"
    return headers


def refs(cve):
    references = cve.get("references") or []
    if isinstance(references, dict):
        references = references.get("referenceData") or []
    for item in references:
        yield item.get("url") or "", item.get("tags") or []


def reference_commits(cve):
    found = []
    for url, tags in refs(cve):
        match = COMMIT_RE.search(url)
        if not match:
            continue
        owner, repo, sha = match.groups()
        score = 0 if {"Patch", "Vendor Advisory", "Release Notes"} & set(tags) else 1
        found.append((score, owner, repo.rstrip("/"), sha))
    for _, owner, repo, sha in sorted(found):
        yield owner, repo, sha


def search_commits(cve_id, token, max_items):
    headers = github_search_headers(token)
    query = quote(f'"{cve_id}"')
    url = f"https://api.github.com/search/commits?q={query}&per_page={max_items}"
    try:
        data = request_json(url, headers=headers)
    except requests.HTTPError:
        return []
    commits = []
    for item in data.get("items") or []:
        html = item.get("html_url") or ""
        match = COMMIT_RE.search(html)
        if match:
            commits.append(match.groups())
    return commits


def iter_github_advisory_cves(start, end, token, limit):
    yielded = 0
    page = 1
    headers = github_headers(token)
    while True:
        data = request_json(
            "https://api.github.com/advisories",
            headers=headers,
            params={
                "type": "reviewed",
                "ecosystem": "pip",
                "published": f"{start}..{end}",
                "per_page": 100,
                "page": page,
            },
        )
        if not data:
            return
        for item in data:
            cve_id = item.get("cve_id")
            if not cve_id:
                continue
            yielded += 1
            yield cve_id
            if limit and yielded >= limit:
                return
        page += 1


def iter_github_advisories(start, end, token, limit):
    yielded = 0
    page = 1
    headers = github_headers(token)
    while True:
        data = request_json(
            "https://api.github.com/advisories",
            headers=headers,
            params={
                "type": "reviewed",
                "ecosystem": "pip",
                "published": f"{start}..{end}",
                "per_page": 100,
                "page": page,
            },
        )
        if not data:
            return
        for item in data:
            yielded += 1
            yield item
            if limit and yielded >= limit:
                return
        page += 1


def iter_github_commit_search(start, end, token, max_pages):
    headers = github_search_headers(token)
    page = 1
    query = f'"CVE-2026" committer-date:>={start} committer-date:<={end}'
    while True:
        data = request_json(
            "https://api.github.com/search/commits",
            headers=headers,
            params={"q": query, "per_page": 100, "page": page},
        )
        items = data.get("items") or []
        if not items:
            return
        for item in items:
            match = COMMIT_RE.search(item.get("html_url") or "")
            if not match:
                continue
            message = ((item.get("commit") or {}).get("message") or "")
            cve_match = CVE_RE.search(message)
            cve_id = cve_match.group(0).upper() if cve_match else "CVE-2026-UNKNOWN"
            owner, repo, sha = match.groups()
            yield cve_id, owner, repo, sha
        page += 1
        if max_pages and page > max_pages:
            return


def get_nvd_cve(cve_id):
    data = request_json(
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        params={"cveId": cve_id},
        sleep=0.6,
    )
    rows = data.get("vulnerabilities") or []
    if not rows:
        return None
    return rows[0].get("cve") or None


def parse_changed_old_lines(patch):
    changed = []
    old_line = None
    for line in (patch or "").splitlines():
        header = HUNK_RE.match(line)
        if header:
            old_line = int(header.group(1))
            changed.append(old_line)
            continue
        if old_line is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            changed.append(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            continue
        else:
            old_line += 1
    return sorted(set(changed))


def old_file_text(owner, repo, path, ref, token):
    headers = github_headers(token)
    api_path = quote(path, safe="")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{api_path}?ref={ref}"
    try:
        data = request_json(url, headers=headers)
        content = data.get("content") or ""
        if content:
            return base64.b64decode(content).decode("utf-8", errors="replace")
    except requests.HTTPError:
        pass
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    return request_text(raw, headers=headers)


def all_functions(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and hasattr(node, "end_lineno"):
            funcs.append(node)
    return funcs


def containing_function(source, changed_lines):
    lines = source.splitlines()
    candidates = []
    for fn in all_functions(source):
        if any(fn.lineno <= line <= fn.end_lineno for line in changed_lines):
            candidates.append(fn)
    if not candidates:
        return None
    fn = min(candidates, key=lambda item: item.end_lineno - item.lineno)
    code = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
    return fn.name, code


def make_prompt(code, ratio):
    lines = code.splitlines()
    if len(lines) < 4:
        return None
    cut = max(1, int(len(lines) * ratio))
    if cut >= len(lines):
        cut = len(lines) - 1
    prompt = "\n".join(lines[:cut]).rstrip() + "\n"
    if len(prompt.strip()) < 20:
        return None
    return prompt


def version_value(value):
    try:
        return Version(str(value))
    except InvalidVersion:
        return None


def pypi_json(package):
    return request_json(f"https://pypi.org/pypi/{package}/json", sleep=0.1)


def release_file(release_files):
    for packagetype in ("sdist", "bdist_wheel"):
        for item in release_files:
            if item.get("packagetype") == packagetype and item.get("url"):
                return item
    return None


def previous_version(releases, patched):
    patched_v = version_value(patched)
    if patched_v is None:
        return None
    candidates = []
    for value, files in releases.items():
        current = version_value(value)
        if current is None or current >= patched_v or not files:
            continue
        if release_file(files) is not None:
            candidates.append((current, value))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def read_archive_py_files(blob, filename):
    files = {}
    if filename.endswith(".whl") or filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                if name.endswith(".py") and not name.endswith("__init__.py"):
                    files[strip_archive_root(name)] = archive.read(name).decode("utf-8", errors="replace")
        return files
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".py") and not member.name.endswith("__init__.py"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                files[strip_archive_root(member.name)] = extracted.read().decode("utf-8", errors="replace")
    return files


def strip_archive_root(path):
    parts = path.split("/")
    if len(parts) > 1:
        return "/".join(parts[1:])
    return path


def download_release_files(package, version):
    data = pypi_json(package)
    files = data.get("releases", {}).get(version) or []
    selected = release_file(files)
    if selected is None:
        return {}, data
    blob = request_bytes(selected["url"], sleep=0.1)
    return read_archive_py_files(blob, selected.get("filename") or ""), data


def changed_old_lines(old_source, new_source):
    old_lines = old_source.splitlines()
    new_lines = new_source.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    changed = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.extend(range(i1 + 1, max(i2, i1 + 1) + 1))
    return changed


def advisory_rows(advisory, args, seen):
    cve_id = advisory.get("cve_id") or (advisory.get("ghsa_id") or "GHSA-Unknown")
    cwes = [item.get("cwe_id") for item in advisory.get("cwes") or [] if item.get("cwe_id")]
    rows = []
    for vuln in advisory.get("vulnerabilities") or []:
        package = ((vuln.get("package") or {}).get("name") or "").strip()
        patched = vuln.get("first_patched_version")
        if isinstance(patched, dict):
            patched = patched.get("identifier")
        if not package or not patched:
            continue
        try:
            patched_files, package_json = download_release_files(package, patched)
        except (requests.HTTPError, tarfile.TarError, zipfile.BadZipFile):
            continue
        old_version = previous_version(package_json.get("releases") or {}, patched)
        if old_version is None:
            continue
        try:
            old_files, _ = download_release_files(package, old_version)
        except (requests.HTTPError, tarfile.TarError, zipfile.BadZipFile):
            continue
        for path, old_source in old_files.items():
            new_source = patched_files.get(path)
            if new_source is None or new_source == old_source:
                continue
            changed = changed_old_lines(old_source, new_source)
            result = containing_function(old_source, changed)
            if result is None:
                continue
            func_name, code = result
            prompt = make_prompt(code, args.prompt_ratio)
            if prompt is None:
                continue
            clean_path = re.sub(r"[^0-9A-Za-z_.-]+", "_", path)
            row_id = f"{cve_id}__pypi_{package}__{old_version}_to_{patched}__{clean_path}__{func_name}.py"
            if row_id in seen:
                continue
            seen.add(row_id)
            rows.append(
                {
                    "ID": row_id,
                    "Prompt": prompt,
                    "Insecure_code": code,
                    "CWE": cwes[0] if cwes else "CWE-Unknown",
                    "CWEs": cwes,
                    "source": {
                        "advisory": advisory.get("ghsa_id"),
                        "cve": advisory.get("cve_id"),
                        "package": package,
                        "old_version": old_version,
                        "patched_version": patched,
                        "file": path,
                        "function": func_name,
                    },
                }
            )
    return rows


def commit_rows(cve_id, cwes, owner, repo, sha, token, prompt_ratio, seen):
    headers = github_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    try:
        data = request_json(url, headers=headers)
    except requests.HTTPError:
        return []
    parents = data.get("parents") or []
    if not parents:
        return []
    parent_sha = parents[0].get("sha")
    rows = []
    for item in data.get("files") or []:
        filename = item.get("filename") or ""
        if not filename.endswith(".py"):
            continue
        changed_lines = parse_changed_old_lines(item.get("patch") or "")
        if not changed_lines:
            continue
        try:
            source = old_file_text(owner, repo, filename, parent_sha, token)
        except (requests.HTTPError, UnicodeDecodeError):
            continue
        result = containing_function(source, changed_lines)
        if result is None:
            continue
        func_name, code = result
        prompt = make_prompt(code, prompt_ratio)
        if prompt is None:
            continue
        clean_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", filename)
        row_id = f"{cve_id}__{owner}_{repo}__{sha[:7]}__{clean_name}__{func_name}.py"
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append(
            {
                "ID": row_id,
                "Prompt": prompt,
                "Insecure_code": code,
                "CWE": cwes[0] if cwes else "CWE-Unknown",
                "CWEs": cwes,
                "source": {
                    "cve": cve_id,
                    "repo": f"{owner}/{repo}",
                    "commit": sha,
                    "parent": parent_sha,
                    "file": filename,
                    "function": func_name,
                },
            }
        )
    return rows


def build(args):
    token = os.environ.get("GITHUB_TOKEN", "")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_rows = set()
    scanned_cves = 0
    scanned_commits = 0
    skipped_cwe = 0
    if args.source == "pypi_advisory":
        for advisory in iter_github_advisories(args.start, args.end, token, args.limit_nvd):
            scanned_cves += 1
            rows.extend(advisory_rows(advisory, args, seen_rows))
            if args.verbose:
                print(
                    f"scanned_advisories={scanned_cves} rows={len(rows)} last={advisory.get('ghsa_id')}",
                    flush=True,
                )
            if len(rows) >= args.target:
                rows = rows[: args.target]
                write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe)
                return rows
        write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe)
        return rows

    if args.source == "github_advisory":
        cve_iter = (get_nvd_cve(cve_id) for cve_id in iter_github_advisory_cves(args.start, args.end, token, args.limit_nvd))
    elif args.source == "nvd":
        cve_iter = iter_nvd_cves(args.start, args.end, args.limit_nvd)
    else:
        cve_iter = None

    if cve_iter is not None:
        for cve in cve_iter:
            if cve is None:
                continue
            cve_id = cve.get("id") or ""
            if not cve_id:
                continue
            scanned_cves += 1
            cwes = cwe_ids(cve)
            if args.cwe_filter == "bandit" and not (set(cwes) & BANDIT_CWES):
                skipped_cwe += 1
                continue
            commits = list(reference_commits(cve))
            if args.search_fallback and len(commits) < args.max_commits_per_cve:
                commits.extend(search_commits(cve_id, token, args.max_commits_per_cve - len(commits)))
            rows, scanned_commits = consume_commits(
                commits, cve_id, cwes, token, args, rows, seen_rows, scanned_cves, scanned_commits, skipped_cwe, output
            )
            if len(rows) >= args.target:
                return rows
        write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe)
        return rows

    seen_commits_global = set()
    for cve_id, owner, repo, sha in iter_github_commit_search(args.start, args.end, token, args.max_search_pages):
        scanned_cves += 1
        key = (owner, repo, sha)
        if key in seen_commits_global:
            continue
        seen_commits_global.add(key)
        rows, scanned_commits = consume_commits(
            [(owner, repo, sha)],
            cve_id,
            ["CWE-Unknown"],
            token,
            args,
            rows,
            seen_rows,
            scanned_cves,
            scanned_commits,
            skipped_cwe,
            output,
        )
        if len(rows) >= args.target:
            return rows
    write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe)
    return rows


def consume_commits(commits, cve_id, cwes, token, args, rows, seen_rows, scanned_cves, scanned_commits, skipped_cwe, output):
    seen_commits = set()
    for owner, repo, sha in commits:
        key = (owner, repo, sha)
        if key in seen_commits:
            continue
        seen_commits.add(key)
        scanned_commits += 1
        rows.extend(commit_rows(cve_id, cwes, owner, repo, sha, token, args.prompt_ratio, seen_rows))
        if args.verbose:
            print(
                f"scanned_cves={scanned_cves} scanned_commits={scanned_commits} rows={len(rows)} last={cve_id}",
                flush=True,
            )
        if len(rows) >= args.target:
            rows = rows[: args.target]
            write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe)
            return rows, scanned_commits
    return rows, scanned_commits


def write_outputs(output, rows, args, scanned_cves, scanned_commits, skipped_cwe):
    dataset = output / "dataset.jsonl"
    with dataset.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "start": args.start,
        "end": args.end,
        "target": args.target,
        "num_rows": len(rows),
        "scanned_cves": scanned_cves,
        "scanned_commits": scanned_commits,
        "skipped_cwe": skipped_cwe,
        "prompt_ratio": args.prompt_ratio,
        "cwe_filter": args.cwe_filter,
        "cwe_counts": Counter(row["CWE"] for row in rows),
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="Unseen2026CVE")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--limit_nvd", type=int, default=0)
    parser.add_argument("--prompt_ratio", type=float, default=0.5)
    parser.add_argument("--max_commits_per_cve", type=int, default=5)
    parser.add_argument("--cwe_filter", choices=["none", "bandit"], default="none")
    parser.add_argument("--source", choices=["nvd", "github_advisory", "github_commit_search", "pypi_advisory"], default="pypi_advisory")
    parser.add_argument("--max_search_pages", type=int, default=10)
    parser.add_argument("--search_fallback", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    today = datetime.now().strftime("%Y-%m-%d")
    if args.end > today:
        args.end = today
    if args.smoke:
        args.target = 5
        if not args.limit_nvd:
            args.limit_nvd = 80
        args.search_fallback = True
        args.verbose = True
    return args


if __name__ == "__main__":
    build(parse_args())
