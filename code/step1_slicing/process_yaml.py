import yaml
import traceback
import os
import sys
import types
import tempfile
import re
import uuid
import shutil
import signal
import argparse
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLICING_PKG_DIR = os.path.join(SCRIPT_DIR, 'slicing')

for _mod_name, _cls_name in [('git', 'Repo')]:
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        setattr(_m, _cls_name, type(_cls_name, (), {}))
        sys.modules[_mod_name] = _m

_slicing_pkg = types.ModuleType('slicing')
_slicing_pkg.__path__ = [SLICING_PKG_DIR]
_slicing_pkg.__package__ = 'slicing'
sys.modules['slicing'] = _slicing_pkg

from slicing.project import Project
from slicing.python.language import PYTHON


def get_all_function_definition_lines(code):
    lines = code.split('\n')
    definition_lines = set()
    paren_count = 0
    in_function_def = False
    pending_decorator_lines = []

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()

        if stripped.startswith('@'):
            pending_decorator_lines.append(line_num)
            continue

        if not stripped:
            continue

        if stripped.startswith('def ') or stripped.startswith('async def '):
            definition_lines.update(pending_decorator_lines)
            pending_decorator_lines = []
            in_function_def = True
            paren_count = 0
        else:
            pending_decorator_lines = []

        if in_function_def:
            definition_lines.add(line_num)
            paren_count += line.count('(') - line.count(')')

            if paren_count == 0 and ':' in line:
                code_part = line.split('#')[0].rstrip()
                if code_part.endswith(':'):
                    in_function_def = False

    return definition_lines


def do_slice(code, slice_lines):
    random_UUID = str(uuid.uuid4())
    tmp_dir = os.path.join(tempfile.gettempdir(), 'slicing_tmp', random_UUID)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, f'{random_UUID}.py'), 'w') as f:
            f.write(code)
        project = Project.create(tmp_dir, language=PYTHON, enable_lsp=False)
        file = project.files[f"{random_UUID}.py"]
        if len(file.functions) > 0:
            function = file.functions[0]
            sliced_statements = function.slice_by_lines(
                lines=slice_lines, control_depth=0, data_dependent_depth=8, control_dependent_depth=8
            )
            lines_set = set()
            for statement in sliced_statements:
                stmt_str = str(statement)
                matches = re.findall(r'line(\d+)-(\d+)', stmt_str)
                if matches:
                    for start, end in matches:
                        for line in range(int(start), int(end) + 1):
                            lines_set.add(line)
            return slice_lines, sorted(list(lines_set - set(slice_lines)))
        return [], []
    except Exception as e:
        print(f"Error in do_slice: {e}")
        print(f"Error in do_slice: {traceback.format_exc()}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        return [], []


def process_yaml_entry(cve_key, entry):
    try:
        pre_data = entry.get("pre", {})
        post_data = entry.get("post", {})

        pre_code = pre_data.get("code", "")
        post_code = post_data.get("code", "")
        delete_lines = entry.get("delete_lines", [])
        add_lines = entry.get("add_lines", [])

        pre_criteria_lines = sorted(delete_lines) if delete_lines else []
        post_criteria_lines = sorted(add_lines) if add_lines else []

        if pre_code and pre_code.strip():
            code_lines = pre_code.split('\n')
            func_def_lines = get_all_function_definition_lines(pre_code)
            criteria_for_slice = [
                line for line in pre_criteria_lines
                if line not in func_def_lines and line <= len(code_lines) and code_lines[line-1].strip()]
            if criteria_for_slice:
                _, pre_relative_lines = do_slice(pre_code, criteria_for_slice)
            else:
                _, pre_relative_lines = [], []
        else:
            _, pre_relative_lines = [], []

        if post_code and post_code.strip():
            code_lines = post_code.split('\n')
            func_def_lines = get_all_function_definition_lines(post_code)
            criteria_for_slice = [
                line for line in post_criteria_lines
                if line not in func_def_lines and line <= len(code_lines) and code_lines[line-1].strip()]
            if criteria_for_slice:
                _, post_relative_lines = do_slice(post_code, criteria_for_slice)
            else:
                _, post_relative_lines = [], []
        else:
            _, post_relative_lines = [], []

        if add_lines and not delete_lines:
            add_set = set(add_lines)
            sorted_add = sorted(list(add_set))
            pure_add_lines = [
                line - sum(1 for a in sorted_add if a < line)
                for line in post_relative_lines
                if line not in add_set
            ]
            pre_relative_lines = pure_add_lines

        if delete_lines and not add_lines:
            delete_set = set(delete_lines)
            sorted_delete = sorted(list(delete_set))
            pure_delete_lines = [
                line - sum(1 for d in sorted_delete if d < line)
                for line in pre_relative_lines
                if line not in delete_set
            ]
            post_relative_lines = pure_delete_lines

        result = {
            "pre": {
                "code": pre_code,
                "file_name_in_cve": pre_data.get("file_name_in_cve"),
                "function_name": pre_data.get("function_name"),
                "label": pre_data.get("label", -1)
            },
            "post": {
                "code": post_code,
                "file_name_in_cve": post_data.get("file_name_in_cve"),
                "function_name": post_data.get("function_name"),
                "label": post_data.get("label", -1)
            },
            "add_lines": add_lines if add_lines else [],
            "delete_lines": delete_lines if delete_lines else [],
            "pre-criteria-lines": pre_criteria_lines,
            "pre-relative-lines": pre_relative_lines,
            "post-criteria-lines": post_criteria_lines,
            "post-relative-lines": post_relative_lines
        }

        return result, None
    except Exception as e:
        error_info = {
            "cve_key": cve_key,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return None, error_info


def worker(args):
    cve_key, entry, timeout = args

    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        result, error = process_yaml_entry(cve_key, entry)

        signal.alarm(0)

        if result:
            return ("success", cve_key, result)
        elif error:
            return ("error", cve_key, error)
        else:
            return ("error", cve_key, {"error": "unknown error"})

    except TimeoutError:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ("timeout", cve_key, {
            "error": "timeout",
            "timeout_at": timestamp
        })
    except Exception as e:
        signal.alarm(0)
        return ("error", cve_key, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def main(input_file, output_file, error_file, num_processes=16):
    print("Loading input file...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        print("Input file is empty or invalid")
        return

    total_entries = len(data)
    print(f"Total entries: {total_entries}")

    results = {}
    errors = {}
    processed_count = 0
    error_count = 0

    if os.path.exists(output_file):
        print("Found existing output file, loading processed results...")
        with open(output_file, 'r', encoding='utf-8') as f:
            results = yaml.safe_load(f) or {}
        processed_count = len(results)
        print(f"Already processed: {processed_count} entries")

    if os.path.exists(error_file):
        with open(error_file, 'r', encoding='utf-8') as f:
            errors = yaml.safe_load(f) or {}
        error_count = len(errors)

    remaining_keys = [k for k in data.keys() if k not in results and k not in errors]
    print(f"Remaining entries: {len(remaining_keys)}")

    if not remaining_keys:
        print("All entries have been processed!")
        return

    timeout = 300
    tasks = [(cve_key, data[cve_key], timeout) for cve_key in remaining_keys]

    print(f"Using {num_processes} processes...")

    save_interval = 100
    tmp_base = os.path.join(tempfile.gettempdir(), 'slicing_tmp')

    try:
        with Pool(processes=num_processes) as pool:
            for result_tuple in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
                status, cve_key, data_result = result_tuple

                if status == "success":
                    results[cve_key] = data_result
                elif status == "timeout":
                    print(f"\nEntry {cve_key} timed out at {data_result.get('timeout_at', 'unknown')}")
                    errors[cve_key] = data_result
                    error_count += 1
                else:
                    print(f"\nEntry {cve_key} failed: {data_result.get('error', 'unknown')}")
                    errors[cve_key] = data_result
                    error_count += 1

                processed_count += 1

                if processed_count % save_interval == 0:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            yaml.dump(results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                        with open(error_file, 'w', encoding='utf-8') as f:
                            yaml.dump(errors, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                    except Exception as e:
                        print(f"\nError during periodic save: {e}")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            with open(error_file, 'w', encoding='utf-8') as f:
                yaml.dump(errors, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error writing output: {e}")
            print(f"Error details: {traceback.format_exc()}")

        print(f"\nDone!")
        print(f"Successful: {len(results)} entries")
        print(f"Errors/Skipped: {error_count} entries")
        print(f"Results saved to: {output_file}")
        print(f"Errors saved to: {error_file}")

    finally:
        if os.path.exists(tmp_base):
            shutil.rmtree(tmp_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-e', '--error', type=str, required=True)
    parser.add_argument('-n', '--num-processes', type=int, default=16)
    args = parser.parse_args()
    main(args.input, args.output, args.error, num_processes=args.num_processes)
