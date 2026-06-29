"""Main inference entry: read YAML config, run split-then-generate on all records."""
import yaml
import json
import argparse
import os
import signal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from method import infer_record, load_weight_data
from yaml_handler import load_yaml_records


def _load_existing_jsonl(jsonl_path):
    results = {}
    if not os.path.exists(jsonl_path):
        return results
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                cve_name = record.get('cve_name')
                if cve_name:
                    results[cve_name] = record
    return results


def _save_result_jsonl(result, cve_name, output_path):
    result_with_name = {'cve_name': cve_name, **result}
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_with_name, ensure_ascii=False) + '\n')


def _timeout_handler(signum, frame):
    raise TimeoutError()


def _process_record(record, config, timeout_seconds, weight_dict):
    cve_name = record.get('CVE-name', 'Unknown')
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = infer_record(record, config, weight_dict)
        return (cve_name, result, None)
    except TimeoutError:
        return (cve_name, None, f"Timeout ({timeout_seconds}s)")
    except Exception as e:
        import traceback
        return (cve_name, None, str(e) + '\n' + traceback.format_exc())
    finally:
        signal.alarm(0)


def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    output_jsonl = config['OUTPUT_JSONL']
    records = load_yaml_records(config['INPUT_YAML'])
    print(f"Total records loaded: {len(records)}")

    processed_cve_names = set()
    if config.get('USE_CACHE') and os.path.exists(output_jsonl):
        existing = _load_existing_jsonl(output_jsonl)
        if existing:
            processed_cve_names = set(existing.keys())
            print(f"Skipping {len(processed_cve_names)} already processed records")

    records_to_process = [
        r for r in records
        if r.get('CVE-name', 'Unknown') not in processed_cve_names
    ]
    print(f"Records to process: {len(records_to_process)}")

    weight_dict = load_weight_data(config['WEIGHT_FILE'])
    print(f"Weight file loaded: {len(weight_dict)} entries")

    save_interval = config['SAVE_INTERVAL']
    timeout_seconds = config['TIMEOUT_SECONDS']
    processed_count = 0

    with ProcessPoolExecutor(max_workers=save_interval) as executor:
        future_to_info = {}
        for idx, record in enumerate(records_to_process):
            future = executor.submit(_process_record, record, config, timeout_seconds, weight_dict)
            future_to_info[future] = (idx, record)

        results_buffer = []
        with tqdm(total=len(records_to_process), desc="Inference") as pbar:
            for future in as_completed(future_to_info):
                idx, record = future_to_info[future]
                cve_name = record.get('CVE-name', 'Unknown')
                cve_name_result, result, error = future.result()
                if error:
                    print(f"\nError processing {cve_name}: {error}")
                    continue
                results_buffer.append((cve_name_result, result))
                processed_count += 1
                if len(results_buffer) >= save_interval:
                    for cve_n, res in results_buffer:
                        _save_result_jsonl(res, cve_n, output_jsonl)
                    results_buffer = []
                pbar.set_postfix({'current': cve_name[:20], 'buffer': len(results_buffer)})
                pbar.update(1)

        if results_buffer:
            for cve_n, res in results_buffer:
                _save_result_jsonl(res, cve_n, output_jsonl)

    print(f"\nDone. Processed {processed_count} records -> {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split-then-generate inference")
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
