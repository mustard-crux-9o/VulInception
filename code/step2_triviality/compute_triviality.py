"""Compute per-token triviality weights for code in a YAML dataset."""
import argparse
import json
import os
import pickle
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from asttokens import ASTTokens

from semantic import semantic_score, semantic_score_for_span, build_semantic_line_index
from syntactic import syntactic_score, syntactic_score_for_span, build_syntactic_line_index
from idf import build_idf_node_index, idf_score_for_span, calculate_df

BASE_CLOSE_KEYWORDS = {
    "else", "elif", "except", "finally",
    "yield", "yield from", "break", "continue",
    "pass", "raise", "lambda", "import", "from", "assert",
    "global", "nonlocal", "in", "is", "as", "and", "or", "not",
    "match", "case", "if", "for", "while", "try", "with",
    "class", "def", "async", "async for", "async with", "async def"
}
FULL_CLOSE_SYMBOLS = {")", "]", "}", ":", ",", ".", ";"}


def build_line_positions(code):
    lines = code.split('\n')
    positions = [0]
    for line in lines:
        positions.append(positions[-1] + len(line) + 1)
    return positions


def resolve_ast_overlap_features(token_start, token_end, active_ast_tokens):
    idf_weight = 0.0
    idf_source = []
    semantic_weight = 0.0
    semantic_source = []
    syntactic_weight = 0.0
    syntactic_source = []
    next_active = []
    for ast_token in active_ast_tokens:
        if ast_token["abs_end"] <= token_start:
            continue
        next_active.append(ast_token)
        overlap_start = max(token_start, ast_token["abs_start"])
        overlap_end = min(token_end, ast_token["abs_end"])
        if overlap_start >= overlap_end:
            continue
        if ast_token["idf_raw"] > idf_weight:
            idf_weight = ast_token["idf_raw"]
            idf_source = ast_token["idf-source"]
        if ast_token["semantic"] > semantic_weight:
            semantic_weight = ast_token["semantic"]
            semantic_source = ast_token["semantic-source"]
        if ast_token["syntactic"] > syntactic_weight:
            syntactic_weight = ast_token["syntactic"]
            syntactic_source = ast_token["syntactic-source"]
    return next_active, idf_weight, idf_source, semantic_weight, semantic_source, syntactic_weight, syntactic_source


def calculate_ast_token_scores(code, cache_data):
    """Compute IDF/semantic/syntactic scores for every AST source token."""
    sem_scores = semantic_score(code)
    syn_scores = syntactic_score(code)
    sem_line_index = build_semantic_line_index(sem_scores)
    syn_line_index = build_syntactic_line_index(syn_scores)
    idf_node_index = build_idf_node_index(code, cache_data, BASE_CLOSE_KEYWORDS, FULL_CLOSE_SYMBOLS)
    atok = ASTTokens(code, parse=True)
    result = []
    for token in atok.tokens:
        if token.startpos == token.endpos:
            continue
        bare = token.string.strip()
        idf_w, idf_src = idf_score_for_span(idf_node_index, token.startpos, token.endpos, bare, BASE_CLOSE_KEYWORDS, FULL_CLOSE_SYMBOLS)
        sem_w, sem_src = semantic_score_for_span(sem_line_index, token.start[0], token.start[1], token.end[1])
        syn_w, syn_src = syntactic_score_for_span(syn_line_index, token.start[0], token.start[1], token.end[1])
        result.append({
            "name": token.string, "line": token.start[0],
            "start": token.start[1], "end": token.end[1],
            "abs_start": token.startpos, "abs_end": token.endpos,
            "idf_raw": idf_w, "idf": idf_w,
            "semantic": sem_w, "syntactic": syn_w,
            "idf-source": idf_src, "semantic-source": sem_src, "syntactic-source": syn_src
        })
    return result


def map_ast_tokens_to_tokenizer(code, tokenizer, ast_token_scores):
    """Map AST-level scores to sub-word tokenizer tokens. weight = semantic * idf * syntactic."""
    encoding = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    offset_mapping = encoding['offset_mapping']
    token_ids = encoding['input_ids']
    tokens = [code[s:e] for s, e in offset_mapping]
    line_positions = build_line_positions(code)
    ast_token_scores = sorted(ast_token_scores, key=lambda x: x["abs_start"])
    ast_idx = 0
    active_ast_tokens = []
    result = []

    for token, token_id, (token_start, token_end) in zip(tokens, token_ids, offset_mapping):
        if token_start == token_end:
            continue
        token_line = bisect_right(line_positions, token_start)
        if token_line > len(line_positions) - 1:
            token_line = len(line_positions) - 1
        line_start = line_positions[token_line - 1] if token_line > 0 else 0
        token_col_start = token_start - line_start
        token_col_end = token_end - line_start

        while ast_idx < len(ast_token_scores) and ast_token_scores[ast_idx]["abs_start"] < token_end:
            active_ast_tokens.append(ast_token_scores[ast_idx])
            ast_idx += 1

        active_ast_tokens, idf_w, idf_src, sem_w, sem_src, syn_w, syn_src = resolve_ast_overlap_features(
            token_start, token_end, active_ast_tokens
        )
        ld_idf = sem_w * idf_w * syn_w
        result.append({
            "name": token, "ids": token_id, "line": token_line,
            "start": token_col_start, "end": token_col_end,
            "weight": ld_idf, "idf_raw": idf_w, "idf": idf_w,
            "semantic": sem_w, "syntactic": syn_w, "ld": sem_w * syn_w,
            "idf-source": idf_src, "semantic-source": sem_src, "syntactic-source": syn_src
        })
    return result


def calculate_combined_weights(code, tokenizer, cache_data):
    """Top-level: compute triviality weights for all tokenizer tokens in a code snippet."""
    if not isinstance(code, str) or not code.strip():
        return []
    try:
        ast_scores = calculate_ast_token_scores(code, cache_data)
    except Exception:
        return []
    return map_ast_tokens_to_tokenizer(code, tokenizer, ast_scores)


_tokenizer = None
_cache_data = None


def _init_worker(tokenizer_path, cache_path, data_dir=None):
    global _tokenizer, _cache_data
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _tokenizer.pad_token = _tokenizer.eos_token
    if data_dir:
        _cache_data = calculate_df(data_dir, cache_path)
    else:
        with open(cache_path, "rb") as f:
            _cache_data = pickle.load(f)


def _process_single_entry(item):
    cve_name, data = item
    result = {"cve-name": cve_name}
    try:
        pre_code = data.get('pre', {}).get('code', None)
        post_code = data.get('post', {}).get('code', None)
        result['pre-code-weight'] = calculate_combined_weights(pre_code, _tokenizer, _cache_data) if pre_code else []
        result['post-code-weight'] = calculate_combined_weights(post_code, _tokenizer, _cache_data) if post_code else []
    except Exception:
        result['pre-code-weight'] = []
        result['post-code-weight'] = []
    return json.dumps(result) + '\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute triviality weights for YAML dataset")
    parser.add_argument('--input', required=True, help="Input YAML file")
    parser.add_argument('--output', required=True, help="Output JSONL file")
    parser.add_argument('--tokenizer', required=True, help="HuggingFace tokenizer path")
    parser.add_argument('--cache', required=True, help="Path to DF stats pickle file (loaded directly, or built from --data-dir then saved here)")
    parser.add_argument('--data-dir', default=None, help="Directory of .py files to build DF stats from (optional; if omitted, --cache must be a pre-built pkl)")
    parser.add_argument('--workers', type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    if args.data_dir:
        calculate_df(args.data_dir, args.cache)
    elif not os.path.exists(args.cache):
        raise FileNotFoundError(f"Cache file not found: {args.cache}. Provide --data-dir to build it.")

    with open(args.input, 'r') as f:
        yaml_data = yaml.safe_load(f)

    items = list(yaml_data.items())
    results = [None] * len(items)
    chunksize = max(1, len(items) // max(args.workers * 8, 1))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, args.cache, args.data_dir),
    ) as executor:
        for idx, result in enumerate(tqdm(
            executor.map(_process_single_entry, items, chunksize=chunksize),
            total=len(items), desc="Computing triviality"
        )):
            results[idx] = result

    with open(args.output, 'w') as f:
        for r in results:
            f.write(r)

    print(f"Done. Output: {args.output}")
