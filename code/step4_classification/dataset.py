"""Data loading, feature computation, and Dataset/collate utilities for the classifier."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

RICH_STATS_DIM = 22
RICH_PROB_FEATURE_DIM = RICH_STATS_DIM * 3


def is_numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def get_selected_lines(record, prefix, use_code):
    if use_code == "criteria":
        return record.get(f"{prefix}-criteria-lines", [])
    if use_code == "relative":
        return record.get(f"{prefix}-relative-lines", [])
    if use_code == "criteria+relative":
        return sorted(set(
            record.get(f"{prefix}-criteria-lines", []) + record.get(f"{prefix}-relative-lines", [])
        ))
    if use_code == "all":
        return record.get(f"{prefix}-processed-lines", [])
    return []


def standardize_feature_lists(train_features, test_features):
    train_tensor = torch.nan_to_num(torch.stack(train_features), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0, unbiased=False).clamp(min=1e-6)
    norm = lambda f: torch.nan_to_num(((f - mean) / std).float(), nan=0.0, posinf=0.0, neginf=0.0)
    return [norm(f) for f in train_features], [norm(f) for f in test_features]


class ClassifierDataset(Dataset):
    """Simple dataset holding (prob_stats, code_pooled, label) tuples."""
    def __init__(self, prob_stats, code_pooled, labels):
        self.prob_stats = prob_stats
        self.code_pooled = code_pooled
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.prob_stats[idx], self.code_pooled[idx], self.labels[idx]


def classifier_collate_fn(batch):
    prob_stats = torch.stack([item[0] for item in batch])
    code_pooled = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return prob_stats, code_pooled, labels


def filter_record(record, filter_mode):
    if filter_mode == "filter_none":
        return True
    pre = record.get("pre-has-code", False)
    post = record.get("post-has-code", False)
    if filter_mode == "filter_empty_post":
        return bool(post)
    if filter_mode == "filter_empty_pre":
        return bool(pre)
    if filter_mode == "filter_empty_all":
        return bool(pre) and bool(post)
    return True


class ClassifierDataLoader:
    def __init__(self, path, filter_mode, embedder, limit=None, batch_size=256,
                 use_cache=True, use_code="all"):
        self.path = Path(path)
        self.filter_mode = filter_mode
        self.embedder = embedder
        self.limit = limit
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.use_code = use_code
        self.code_dim = embedder.model.config.hidden_size
        self.prob_stats = []
        self.code_pooled = []
        self.labels = []
        self.records = []
        self._load_and_process()

    def _get_cache_path(self):
        path_str = str(self.path.resolve())
        with open(path_str, "r", encoding="utf-8") as f:
            first_1mb = f.read(1024 * 1024)
        limit_str = str(self.limit) if self.limit else "all"
        cache_key = f"statement_prefix_v6_{path_str}_{self.filter_mode}_{limit_str}_{self.use_code}_{self.embedder.model.config._name_or_path}_{first_1mb}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = self.path.parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{cache_hash}.pkl"

    def _load_cache(self):
        if not self.use_cache:
            return False
        cache_path = self._get_cache_path()
        if cache_path.exists():
            print(f"Loading cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.prob_stats = data["prob_stats"]
            self.code_pooled = data["code_pooled"]
            self.labels = data["labels"]
            self.records = data["records"]
            return True
        return False

    def _save_cache(self):
        if not self.use_cache:
            return
        cache_path = self._get_cache_path()
        print(f"Saving cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({"prob_stats": self.prob_stats, "code_pooled": self.code_pooled,
                          "labels": self.labels, "records": self.records}, f)

    def _load_and_process(self):
        if self._load_cache():
            return
        jobs, owners, sums, denominators = [], [], [], []
        with self.path.open("r", encoding="utf-8") as file:
            for idx, line in enumerate(tqdm(file, desc="Loading data", total=self.limit)):
                if self.limit is not None and idx >= self.limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = record.get('pre-label')
                if not is_numeric(label) or not filter_record(record, self.filter_mode):
                    continue
                metadata = record.get('corpus-metadata')
                if record.get('feature-version') != 5 or not isinstance(metadata, dict) or metadata.get('format_version') != 3 or any(
                    f'{side}-statements' not in record or f'{side}-has-code' not in record
                    or f'{side}-processed-lines' not in record
                    or any('prob-stats' not in statement or 'abstracted-prefix' not in statement
                           or 'abstracted-original' not in statement or 'abstracted-generated' not in statement
                           for statement in record[f'{side}-statements'])
                    for side in ('pre', 'post')
                ):
                    raise ValueError(f"Old inference features for {record.get('cve_name', idx)}; rerun step2 inference")
                record_index = len(self.labels)
                accumulators = [np.zeros(self.code_dim, dtype=np.float32) for _ in range(4)]
                side_weights = [0.0, 0.0]
                side_features = []
                for side_index, side in enumerate(('pre', 'post')):
                    selected_lines = get_selected_lines(record, side, self.use_code)
                    selected = set(selected_lines)
                    side_jobs = [statement for statement in record[f'{side}-statements']
                                 if any(line in selected for line in range(
                                     statement['start_line'], statement['end_line'] + 1))]
                    features = [statement['prob-stats'][self.use_code] for statement in side_jobs
                                if statement['prob-stats'][self.use_code] is not None]
                    side_features.append(np.mean(features, axis=0).tolist() if features
                                         else [0.0] * RICH_STATS_DIM)
                    for statement in side_jobs:
                        weight = float(statement['weight'])
                        if weight <= 0:
                            continue
                        side_weights[side_index] += weight
                        prefix = statement['abstracted-prefix']
                        jobs.append((prefix, statement['abstracted-original']))
                        owners.append((record_index, side_index * 2, weight))
                        generated = statement['abstracted-generated']
                        if generated is not None:
                            jobs.append((prefix, generated))
                            owners.append((record_index, side_index * 2 + 1, weight))
                pre, post = map(np.asarray, side_features)
                self.prob_stats.append(torch.tensor(np.concatenate((pre, post, pre - post)), dtype=torch.float32))
                sums.append(accumulators)
                denominators.append(side_weights)
                self.labels.append(float(label))
                self.records.append(record)
        if not self.records:
            raise ValueError('No valid samples found')
        vectors, truncated = self.embedder.get_prefix_statement_embeddings(jobs, batch_size=min(self.batch_size, 64))
        for vector, (record_index, position, weight) in zip(vectors, owners):
            sums[record_index][position] += vector * weight
        for i in range(len(self.labels)):
            for side in (0, 1):
                denominator = denominators[i][side]
                if denominator:
                    sums[i][side * 2] /= denominator
                    sums[i][side * 2 + 1] /= denominator
            self.code_pooled.append(torch.tensor(np.concatenate(sums[i]), dtype=torch.float32))
        failed = sum(s.get('generated-status') not in (None, 'ok')
                     for r in self.records for side in ('pre', 'post') for s in r[f'{side}-statements'])
        zero_weight = sum(s['weight'] <= 0 for r in self.records for side in ('pre', 'post')
                          for s in r[f'{side}-statements'])
        print(f"Prefix embeddings: {len(jobs)} encoded, {failed} invalid generations, "
              f"{truncated} truncated statements, {zero_weight} zero-weight statements")
        self._save_cache()
