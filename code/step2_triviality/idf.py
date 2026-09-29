import ast
import hashlib
import os
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from asttokens import ASTTokens

CORPUS_CACHE_VERSION = 3


def _corpus_files(data_dir):
    return sorted(os.path.join(root, file) for root, _, files in os.walk(data_dir)
                  for file in files if file.endswith('.py'))


def _manifest_entry(digest, root, path, content_hash):
    relative = os.path.relpath(path, root).replace(os.sep, '/')
    digest.update(relative.encode('utf-8', 'surrogateescape') + b'\0' +
                  content_hash.encode('ascii') + b'\n')


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus_source(cache_data, data_dir):
    """Verify a cached DF table against the bytes of its declared source files."""
    root = os.path.realpath(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Corpus directory not found: {root}')
    files = _corpus_files(root)
    digest = hashlib.sha256()
    for path in tqdm(files, desc='Verifying corpus provenance'):
        _manifest_entry(digest, root, path, _sha256_file(path))
    meta = cache_data['corpus_metadata']
    if len(files) != meta['candidate_files'] or digest.hexdigest() != meta['manifest_sha256']:
        raise ValueError(f'Corpus files differ from DF cache provenance: {root}')


def load_corpus_cache(cache_file):
    with open(cache_file, 'rb') as source:
        cache_data = pickle.load(source)
    meta = cache_data.get('corpus_metadata')
    if not isinstance(meta, dict) or meta.get('format_version') != CORPUS_CACHE_VERSION:
        raise ValueError(f'DF cache has no verifiable corpus provenance: {cache_file}; '
                         'rebuild it from --data-dir into a new cache file')
    if meta.get('df_key_schema') != 'ast-call-name-v1' or not meta.get('manifest_sha256'):
        raise ValueError(f'Unsupported DF key schema or missing source manifest: {cache_file}')
    if meta.get('parsed_files') != cache_data.get('total_files') or cache_data['total_files'] <= 0:
        raise ValueError(f'Invalid DF cache file count: {cache_file}')
    return cache_data


def get_node_key(node):
    """Map an AST node to a (name, category) key for DF counting."""
    t = type(node)
    if t is ast.Name:
        return (node.id, "Name")
    if t is ast.FunctionDef:
        return (node.name, "FunctionDef")
    if t is ast.AsyncFunctionDef:
        return (node.name, "AsyncFunctionDef")
    if t is ast.ClassDef:
        return (node.name, "ClassDef")
    if t is ast.Call:
        f = node.func
        ft = type(f)
        if ft is ast.Name:
            return (f.id, "Call")
        if ft is ast.Attribute:
            return (f.attr, "Call.Attribute")
        return ("Call", "Call")
    if t is ast.Attribute:
        return (node.attr, "Attribute")
    if t is ast.Subscript:
        s = node.slice
        if type(s) is ast.Index:
            s = s.value
        st = type(s)
        if st is ast.Name:
            return (s.id, "Subscript")
        if st is ast.Constant:
            return (str(s.value), "Subscript")
        return ("Subscript", "Subscript")
    if t is ast.List:
        return ("List", "List")
    if t is ast.Dict:
        return ("Dict", "Dict")
    if t is ast.Tuple:
        return ("Tuple", "Tuple")
    if t is ast.Constant:
        return (str(node.value), "Constant")
    return None


def _process_file(filepath):
    with open(filepath, 'rb') as source:
        raw = source.read()
    content_hash = hashlib.sha256(raw).hexdigest()
    code = raw.decode('utf-8', errors='ignore')
    try:
        if code.count("\n") + 1 > 1000:
            return filepath, content_hash, None, 'too_long'
        tree = ast.parse(code)
        file_nodes = defaultdict(int)
        for node in ast.walk(tree):
            key = get_node_key(node)
            if key:
                file_nodes[key] += 1
        return filepath, content_hash, dict(file_nodes), 'parsed'
    except Exception:
        return filepath, content_hash, None, 'parse_error'


def calculate_df(data_dir, cache_file, corpus_id=None):
    """Build or verify type-conditioned DF counts and their source manifest."""
    if os.path.exists(cache_file):
        cache_data = load_corpus_cache(cache_file)
        if corpus_id is not None and cache_data['corpus_metadata']['declared_source'] != corpus_id:
            raise ValueError(f'Declared corpus differs from DF cache: {corpus_id}')
        verify_corpus_source(cache_data, data_dir)
        return cache_data
    root = os.path.realpath(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f'Corpus directory not found: {root}')
    all_files = _corpus_files(root)
    cve_named_candidates = sum(
        os.path.relpath(path, root).split(os.sep, 1)[0].startswith('CVE-')
        for path in all_files
    )
    node_stats = defaultdict(lambda: {"files": set(), "total_count": 0})
    total_files = 0
    cve_named_parsed = 0
    skipped = defaultdict(int)
    manifest = hashlib.sha256()
    with Pool(8) as pool:
        for filepath, content_hash, file_nodes, status in tqdm(
                pool.imap(_process_file, all_files), total=len(all_files), desc='Building DF stats'):
            _manifest_entry(manifest, root, filepath, content_hash)
            if file_nodes is not None:
                total_files += 1
                if os.path.relpath(filepath, root).split(os.sep, 1)[0].startswith('CVE-'):
                    cve_named_parsed += 1
                for key, count in file_nodes.items():
                    node_stats[key]["files"].add(filepath)
                    node_stats[key]["total_count"] += count
            else:
                skipped[status] += 1
    if total_files == 0:
        raise ValueError(f'No parseable Python files in reference corpus: {root}')
    result = {
        "node_stats": {k: {"df_obs": len(v["files"]), "total_count": v["total_count"]} for k, v in node_stats.items()},
        "total_files": total_files,
        "corpus_metadata": {
            "format_version": CORPUS_CACHE_VERSION,
            "declared_source": corpus_id or os.path.basename(os.path.dirname(root)),
            "source_root": root,
            "manifest_sha256": manifest.hexdigest(),
            "candidate_files": len(all_files),
            "parsed_files": total_files,
            "cve_named_candidate_files": cve_named_candidates,
            "cve_named_parsed_files": cve_named_parsed,
            "skipped_too_long": skipped['too_long'],
            "skipped_parse_error": skipped['parse_error'],
            "source_note": ("Local source tree including CVE-* directories; upstream archive revision unverified"
                            if cve_named_candidates else
                            "Local source tree; upstream archive revision unverified"),
            "selection": "recursive .py; <=1000 physical lines; UTF-8 errors=ignore; ast.parse",
            "python_version": sys.version.split()[0],
            "df_key_schema": "ast-call-name-v1"
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def inverse_document_frequency(code, cache_data):
    """Paper Eq. (4): type-conditioned corpus non-triviality, 1 - df/N."""
    node_stats = cache_data["node_stats"]
    total_files = cache_data["total_files"]
    if total_files <= 0:
        raise ValueError("Corpus DF cache has no parsed files")
    tree = ast.parse(code)
    code_nodes = set()
    for node in ast.walk(tree):
        key = get_node_key(node)
        if key is not None:
            code_nodes.add(key)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            code_nodes.add((node.arg, "Name"))
    idf_scores = {}
    for key in code_nodes:
        df_obs = node_stats[key]["df_obs"] if key in node_stats else 0
        idf_scores[key] = 1.0 - min(df_obs, total_files) / total_files
    return idf_scores


def _get_node_literal_range(node, atok):
    try:
        if isinstance(node, ast.Name):
            return atok.get_text_range(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            full_start, _ = atok.get_text_range(node)
            idx = atok.text.find(node.name, full_start)
            if idx != -1:
                return idx, idx + len(node.name)
        if isinstance(node, ast.Attribute):
            full_start, full_end = atok.get_text_range(node)
            idx = atok.text.rfind(node.attr, full_start, full_end)
            if idx != -1:
                return idx, idx + len(node.attr)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return atok.get_text_range(node.func)
            if isinstance(node.func, ast.Attribute):
                return _get_node_literal_range(node.func, atok)
        if isinstance(node, ast.Constant):
            return atok.get_text_range(node)
    except Exception:
        pass
    return None, None


def build_idf_node_index(code, cache_data, base_close_keywords=None, full_close_symbols=None):
    """Build sorted (abs_start, abs_end, idf, key_str) index for a code snippet."""
    if base_close_keywords is None:
        base_close_keywords = set()
    if full_close_symbols is None:
        full_close_symbols = set()
    idf_scores = inverse_document_frequency(code, cache_data)
    atok = ASTTokens(code, parse=True)
    tree = atok.tree
    lines = code.split("\n")
    line_positions = [0]
    for i in range(len(lines)):
        line_positions.append(line_positions[-1] + len(lines[i]) + 1)
    node_index = []
    for node in ast.walk(tree):
        key = get_node_key(node)
        if key is None or key == (None, None) or key not in idf_scores:
            continue
        node_start, node_end = _get_node_literal_range(node, atok)
        if node_start is None:
            continue
        node_text = code[node_start:node_end]
        has_keyword_or_symbol = node_text in base_close_keywords or node_text in full_close_symbols
        node_idf = 0.0 if has_keyword_or_symbol else idf_scores[key]
        node_index.append((node_start, node_end, node_idf, str(key)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.arg):
            continue
        key = (node.arg, "Name")
        if key not in idf_scores:
            continue
        if not (hasattr(node, "lineno") and hasattr(node, "col_offset")):
            continue
        node_start = line_positions[node.lineno - 1] + node.col_offset
        node_end = node_start + len(node.arg)
        node_text = code[node_start:node_end]
        has_keyword_or_symbol = node_text in base_close_keywords or node_text in full_close_symbols
        node_idf = 0.0 if has_keyword_or_symbol else idf_scores[key]
        node_index.append((node_start, node_end, node_idf, str(key)))
    node_index.sort(key=lambda x: x[0])
    return node_index


def idf_score_for_span(node_index, token_start, token_end, token_bare, base_close_keywords=None, full_close_symbols=None):
    """Return the max IDF score overlapping a given character span."""
    if base_close_keywords is None:
        base_close_keywords = set()
    if full_close_symbols is None:
        full_close_symbols = set()
    idf_weight = 0.0
    idf_source = []
    if token_bare in base_close_keywords or token_bare in full_close_symbols:
        return idf_weight, idf_source
    for node_start, node_end, node_idf, key_str in node_index:
        if node_start >= token_end:
            break
        if node_end <= token_start:
            continue
        overlap_start = max(token_start, node_start)
        overlap_end = min(token_end, node_end)
        if overlap_start < overlap_end and node_idf > idf_weight:
            idf_weight = node_idf
            idf_source = [key_str]
    return idf_weight, idf_source
