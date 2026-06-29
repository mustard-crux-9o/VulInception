"""Data loading, feature computation, and Dataset/collate utilities for the classifier."""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MAX_LENGTH = 4096
RICH_STATS_DIM = 22
RICH_PROB_FEATURE_DIM = RICH_STATS_DIM * 3


def is_numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def flatten_numeric(seq):
    flat = []
    def collect(v):
        if is_numeric(v):
            flat.append(float(v))
        elif isinstance(v, list):
            for item in v:
                collect(item)
    collect(seq)
    return flat


def extract_lines_from_code(code, line_numbers):
    if not code or not isinstance(code, str) or not line_numbers:
        return code if isinstance(code, str) else ""
    lines = code.split("\n")
    return "\n".join(lines[ln - 1] for ln in sorted(set(line_numbers)) if 0 <= ln - 1 < len(lines))


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
        return [e[0] for e in record.get(f"{prefix}-split_offset", [])]
    return []


def align_feature_pair(primary, secondary, primary_name="primary", secondary_name="secondary", context=""):
    """Assert two feature lists have the same length."""
    msg = f"{primary_name} length {len(primary)} != {secondary_name} length {len(secondary)}"
    if context:
        msg = f"{context}: {msg}"
    assert len(primary) == len(secondary), msg
    return primary, secondary


def sanitize_numeric_sequence(values):
    sanitized = []
    for v in values:
        fv = float(v)
        if not np.isfinite(fv):
            fv = 0.0
        sanitized.append(fv)
    return sanitized


def align_to_longer(seq_a, seq_b):
    list_a = sanitize_numeric_sequence(seq_a)
    list_b = sanitize_numeric_sequence(seq_b)
    if not list_a and not list_b:
        return [], []
    max_len = max(len(list_a), len(list_b))
    list_a += [0.0] * (max_len - len(list_a))
    list_b += [0.0] * (max_len - len(list_b))
    return list_a, list_b


def safe_corrcoef(seq_a, seq_b):
    if not seq_a or not seq_b:
        return 0.0
    a, b = align_to_longer(seq_a, seq_b)
    if len(a) < 2:
        return 0.0
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if float(arr_a.std()) < 1e-8 or float(arr_b.std()) < 1e-8:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def compute_prob_stats(probs):
    """Compute 7-dim summary statistics for a probability sequence."""
    if not probs:
        return [0.0] * 7
    arr = np.array(probs)
    return [float(np.mean(arr)), float(np.max(arr)), float(np.min(arr)),
            float(np.median(arr)), float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)), float(np.std(arr))]


def compute_rich_stats(values):
    """Compute 22-dim rich statistics for a numeric sequence."""
    if not values:
        return [0.0] * RICH_STATS_DIM
    arr = np.asarray(sanitize_numeric_sequence(values), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(arr.mean())
    std = float(arr.std())
    max_v = float(arr.max())
    min_v = float(arr.min())
    median = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    first = float(arr[0])
    last = float(arr[-1])
    energy = float(np.mean(arr ** 2))
    top_k = float(np.mean(np.sort(arr)[-min(3, arr.size):]))
    bottom_k = float(np.mean(np.sort(arr)[:min(3, arr.size)]))
    high_ratio = float(np.mean(arr >= 0.9))
    low_ratio = float(np.mean(arr <= 0.1))
    if arr.size > 1:
        diff = np.diff(arr)
        diff_mean, diff_std = float(diff.mean()), float(diff.std())
    else:
        diff_mean, diff_std = 0.0, 0.0
    centered = arr - mean
    skew = float(np.mean(centered ** 3) / ((std ** 3) + 1e-8))
    kurt = float(np.mean(centered ** 4) / ((std ** 4) + 1e-8))
    return [float(arr.size), mean, max_v, min_v, median, p25, p75, std,
            first, last, last - first, max_v - min_v, p75 - p25, energy,
            top_k, bottom_k, high_ratio, low_ratio, diff_mean, diff_std, skew, kurt]


def prepare_prob_inputs(pre_prob, pre_triv, post_prob, post_triv, sample_name="sample"):
    """Sanitize, align, and truncate prob/triviality sequences."""
    pre_prob = sanitize_numeric_sequence(pre_prob)
    pre_triv = sanitize_numeric_sequence(pre_triv)
    post_prob = sanitize_numeric_sequence(post_prob)
    post_triv = sanitize_numeric_sequence(post_triv)
    pre_prob, pre_triv = align_feature_pair(list(pre_prob), list(pre_triv),
                                             "pre-probs", "pre-triviality", sample_name)
    post_prob, post_triv = align_feature_pair(list(post_prob), list(post_triv),
                                               "post-probs", "post-triviality", sample_name)
    if not pre_prob and post_prob:
        pre_prob = [0.0] * len(post_prob)
        pre_triv = [0.0] * len(post_prob)
    if not post_prob and pre_prob:
        post_prob = [0.0] * len(pre_prob)
        post_triv = [0.0] * len(pre_prob)
    if not pre_prob and not post_prob:
        pre_prob = pre_triv = [0.0] * 100
        post_prob = post_triv = [0.0] * 100
    return pre_prob[:MAX_LENGTH], pre_triv[:MAX_LENGTH], post_prob[:MAX_LENGTH], post_triv[:MAX_LENGTH]


def weight_token_probabilities(probs, triviality, context=""):
    """Compute the paper-defined per-token characteristic p(t) * tau(t)."""
    probs = sanitize_numeric_sequence(probs)
    triviality = sanitize_numeric_sequence(triviality)
    probs, triviality = align_feature_pair(
        probs, triviality, "probabilities", "triviality", context
    )
    return [prob * tau for prob, tau in zip(probs, triviality)]


def compute_rich_features(pre_prob, pre_triv, post_prob, post_triv):
    """Compute 66 features: 22 each for pre, post, and pre-post difference."""
    pre_weighted = weight_token_probabilities(pre_prob, pre_triv, "pre")
    post_weighted = weight_token_probabilities(post_prob, post_triv, "post")
    pre_stats = compute_rich_stats(pre_weighted)
    post_stats = compute_rich_stats(post_weighted)
    diff_stats = [pre_value - post_value for pre_value, post_value in zip(pre_stats, post_stats)]
    features = pre_stats + post_stats + diff_stats
    assert len(features) == RICH_PROB_FEATURE_DIM
    return features


@torch.no_grad()
def triviality_weighted_pooling(embedder, codes, trivs_per_code, batch_size=64):
    hs = embedder.model.config.hidden_size
    results = []
    for start in tqdm(range(0, len(codes), batch_size), desc="TrivWeightedPool"):
        end = min(start + batch_size, len(codes))
        batch_codes = codes[start:end]
        batch_trivs = trivs_per_code[start:end]
        valid = [i for i, c in enumerate(batch_codes) if c and c.strip()]
        if not valid:
            results.extend(np.zeros(hs, dtype=np.float32) for _ in batch_codes)
            continue
        valid_codes = [batch_codes[i] for i in valid]
        enc = embedder.tokenizer(valid_codes, return_tensors="pt", padding=True, truncation=True, max_length=512)
        attn = enc["attention_mask"]
        enc_dev = {k: v.to(embedder.device) for k, v in enc.items()}
        hidden = embedder.model(**enc_dev).last_hidden_state.cpu().numpy()
        attn_np = attn.numpy()
        batch_results = [None] * len(batch_codes)
        for bi, vi in enumerate(valid):
            seq_len = int(attn_np[bi].sum())
            h = hidden[bi, 1:seq_len - 1]
            if h.shape[0] == 0:
                batch_results[vi] = np.zeros(hs, dtype=np.float32)
                continue
            triv_flat = batch_trivs[vi]
            if not triv_flat:
                batch_results[vi] = h.mean(0).astype(np.float32)
                continue
            n_tokens = h.shape[0]
            n_triv = len(triv_flat)
            if n_triv >= n_tokens:
                triv_arr = np.array(triv_flat[:n_tokens], dtype=np.float32)
            else:
                triv_arr = np.zeros(n_tokens, dtype=np.float32)
                triv_arr[:n_triv] = triv_flat
            w = np.clip(1.0 - triv_arr, 0.01, 1.0)
            w = w / (w.sum() + 1e-8)
            batch_results[vi] = (h * w[:, None]).sum(0).astype(np.float32)
        for i in range(len(batch_codes)):
            results.append(batch_results[i] if batch_results[i] is not None else np.zeros(hs, dtype=np.float32))
    return results


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
    pre = record.get("pre-code", "")
    post = record.get("post-code", "")
    if filter_mode == "filter_empty_post":
        return bool(post and post.strip())
    if filter_mode == "filter_empty_pre":
        return bool(pre and pre.strip())
    if filter_mode == "filter_empty_all":
        return bool(pre and pre.strip()) and bool(post and post.strip())
    return True


def get_code_for_embedding(record, use_code="all"):
    pre_code = record.get("pre-code", "")
    gen_pre = record.get("generated-pre-code", "")
    post_code = record.get("post-code", "")
    gen_post = record.get("generated-post-code", "")
    if use_code == "all":
        return pre_code, gen_pre, post_code, gen_post
    for prefix, code_key, gen_key in [("pre", "pre-code", "generated-pre-code"),
                                       ("post", "post-code", "generated-post-code")]:
        sel = get_selected_lines(record, prefix, use_code)
        if sel:
            if prefix == "pre":
                pre_code = extract_lines_from_code(pre_code, sel)
                gen_pre = extract_lines_from_code(gen_pre, sel)
            else:
                post_code = extract_lines_from_code(post_code, sel)
                gen_post = extract_lines_from_code(gen_post, sel)
    return pre_code, gen_pre, post_code, gen_post


def get_triv_flat(record, prefix, use_code):
    so = record.get(f"{prefix}-split_offset", [])
    triv = record.get(f"{prefix}-triviality", [])
    sel = get_selected_lines(record, prefix, use_code)
    if use_code == "all":
        flat = []
        for t in triv:
            flat.extend(flatten_numeric(t))
        return flat
    l2i = {e[0]: j for j, e in enumerate(so)}
    flat = []
    for ln in sel:
        si = l2i.get(ln)
        if si is not None and si < len(triv):
            flat.extend(flatten_numeric(triv[si]))
    return flat


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
        cache_key = f"p_times_tau_66_diff_v1_{path_str}_{self.filter_mode}_{limit_str}_{self.use_code}_{first_1mb}"
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
        pre_codes, gen_pre_codes, post_codes, gen_post_codes = [], [], [], []
        pre_trivs, gen_pre_trivs, post_trivs, gen_post_trivs = [], [], [], []
        prob_stats_list = []

        with self.path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f, desc="Loading data", total=self.limit)):
                if self.limit is not None and idx >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = record.get("pre-label", None)
                if label is None or not is_numeric(label):
                    continue
                label = int(label)
                if not filter_record(record, self.filter_mode):
                    continue
                pre_code, gen_pre, post_code, gen_post = get_code_for_embedding(record, self.use_code)
                pre_prob = flatten_numeric(record.get("pre-probs", []))
                pre_triv = flatten_numeric(record.get("pre-triviality", []))
                post_prob = flatten_numeric(record.get("post-probs", []))
                post_triv = flatten_numeric(record.get("post-triviality", []))
                sample_name = record.get("CVE-name", record.get("cve_name", f"sample-{idx + 1}"))
                pre_prob, pre_triv, post_prob, post_triv = prepare_prob_inputs(
                    pre_prob, pre_triv, post_prob, post_triv, sample_name)
                pre_codes.append(pre_code)
                gen_pre_codes.append(gen_pre)
                post_codes.append(post_code)
                gen_post_codes.append(gen_post)
                pre_trivs.append(get_triv_flat(record, "pre", self.use_code))
                gen_pre_trivs.append(get_triv_flat(record, "pre", self.use_code))
                post_trivs.append(get_triv_flat(record, "post", self.use_code))
                gen_post_trivs.append(get_triv_flat(record, "post", self.use_code))
                prob_stats_list.append(compute_rich_features(pre_prob, pre_triv, post_prob, post_triv))
                self.labels.append(float(label))
                self.records.append(record)

        if not pre_codes:
            raise ValueError("No valid samples found")

        print("Computing triviality-weighted code embeddings...")
        all_pre = triviality_weighted_pooling(self.embedder, pre_codes, pre_trivs, self.batch_size)
        all_gen_pre = triviality_weighted_pooling(self.embedder, gen_pre_codes, gen_pre_trivs, self.batch_size)
        all_post = triviality_weighted_pooling(self.embedder, post_codes, post_trivs, self.batch_size)
        all_gen_post = triviality_weighted_pooling(self.embedder, gen_post_codes, gen_post_trivs, self.batch_size)

        for i in range(len(self.labels)):
            combined = np.concatenate([all_pre[i], all_gen_pre[i], all_post[i], all_gen_post[i]])
            self.code_pooled.append(torch.tensor(combined, dtype=torch.float32))
            self.prob_stats.append(torch.tensor(prob_stats_list[i], dtype=torch.float32))
        self._save_cache()
