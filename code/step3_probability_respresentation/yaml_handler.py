"""Load YAML records into a flat list of dicts for inference."""
import os
import yaml


def load_yaml_records(yaml_path):
    """Convert nested YAML (key=CVE-name) into a list of record dicts."""
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    records = []
    if isinstance(data, dict):
        for cve_name, cve_data in data.items():
            record = {
                "CVE-name": cve_name,
                "pre-code": cve_data.get("pre", {}).get("code", ""),
                "post-code": cve_data.get("post", {}).get("code", ""),
                "pre-label": cve_data.get("pre", {}).get("label", 0),
                "post-label": cve_data.get("post", {}).get("label", 0),
                "pre-criteria-lines": cve_data.get("pre-criteria-lines", []),
                "post-criteria-lines": cve_data.get("post-criteria-lines", []),
                "pre-relative-lines": cve_data.get("pre-relative-lines", []),
                "post-relative-lines": cve_data.get("post-relative-lines", []),
            }
            records.append(record)
    return records
