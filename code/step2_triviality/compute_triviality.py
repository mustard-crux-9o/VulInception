"""Compute per-token triviality weights for code in a YAML dataset."""
import argparse
import hashlib
import json
import os
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from preprocessing.statements import extract_statements

from semantic import semantic_score
from syntactic import syntactic_score, extract_ast_nodes
from idf import calculate_df, load_corpus_cache

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
    active = [node for node in active_ast_tokens if node['abs_end'] > token_start]
    overlapping = [node for node in active if node['abs_start'] < token_end]
    if not overlapping:
        return active, 0.0, [], 1.0, [], 1.0, []
    # One subword inherits one content node's entire product, never three
    # independently maximized factors from three different nodes.
    node = max(overlapping, key=lambda n: (
        min(token_end, n['abs_end']) - max(token_start, n['abs_start']),
        -(n['abs_end'] - n['abs_start']),
        n['type'] in ('Call', 'Call.Attribute')))
    return (active, node['idf_raw'], node['idf-source'], node['semantic'],
            node['semantic-source'], node['syntactic'], node['syntactic-source'])


def calculate_ast_token_scores(code, cache_data):
    """Compute the three paper factors for each source-addressable AST node."""
    sem_scores = semantic_score(code)
    syn_scores = syntactic_score(code)
    tokens, types = extract_ast_nodes(code)
    starts = build_line_positions(code)
    stats = cache_data['node_stats']
    total = cache_data['total_files']
    if total <= 0:
        raise ValueError('Corpus DF cache has no parsed files')
    result = []
    for (line, start, end, name), node_type in zip(tokens, types):
        if end <= start or line < 1 or line > len(code.splitlines()):
            continue
        category = 'Name' if node_type == 'arg' else node_type
        key = (name, category)
        df = stats.get(key, {}).get('df_obs', 0)
        corp = 1.0 - min(df, total) / total
        semantic_key = (line, start, end, name)
        semantic_supported = semantic_key in sem_scores
        sem_w = sem_scores.get(semantic_key, 1.0)
        syn_w = syn_scores.get((line, start, end, name, node_type), 1.0)
        source_line = code.splitlines()[line - 1]
        abs_start = starts[line - 1] + len(source_line.encode('utf-8')[:start].decode('utf-8'))
        abs_end = starts[line - 1] + len(source_line.encode('utf-8')[:end].decode('utf-8'))
        result.append({
            "name": name, "type": node_type, "line": line,
            "start": start, "end": end,
            "abs_start": abs_start, "abs_end": abs_end,
            "idf_raw": corp, "idf": corp,
            "semantic": sem_w, "syntactic": syn_w,
            "semantic-status": ('computed' if semantic_supported else
                                'unsupported-scope' if node_type == 'Name' else 'not-identifier'),
            "weight": corp * sem_w * syn_w,
            "idf-source": [str(key)], "semantic-source": [name] if sem_w != 1 else [],
            "syntactic-source": [name] if syn_w != 1 else []
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
    """Return aligned subword, AST-node, and statement scores."""
    if not isinstance(code, str) or not code.strip():
        return [], [], []
    ast_scores = calculate_ast_token_scores(code, cache_data)
    token_scores = map_ast_tokens_to_tokenizer(code, tokenizer, ast_scores)
    statements = extract_statements(code)
    for stmt in statements:
        nodes = [n for n in ast_scores if stmt['start'] <= n['abs_start'] and n['abs_end'] <= stmt['end']]
        stmt['weight'] = sum(n['weight'] for n in nodes) / len(nodes) if nodes else 0.0
        stmt['node_count'] = len(nodes)
    return token_scores, ast_scores, statements


_tokenizer = None
_cache_data = None
_cache_sha256 = None


def _init_worker(tokenizer_path, cache_path, cache_sha256):
    global _tokenizer, _cache_data, _cache_sha256
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _tokenizer.pad_token = _tokenizer.eos_token
    _cache_data = load_corpus_cache(cache_path)
    _cache_sha256 = cache_sha256


def _process_single_entry(item):
    cve_name, data = item
    result = {"cve-name": cve_name}
    try:
        pre_code = data.get('pre', {}).get('code', None)
        post_code = data.get('post', {}).get('code', None)
        result['feature-version'] = 3
        result['corpus-metadata'] = {
            **_cache_data['corpus_metadata'], 'cache_sha256': _cache_sha256
        }
        for side, code in [('pre', pre_code), ('post', post_code)]:
            weights, nodes, statements = calculate_combined_weights(code, _tokenizer, _cache_data)
            result[f'{side}-code-weight'] = weights
            result[f'{side}-node-scores'] = nodes
            result[f'{side}-statements'] = statements
            result[f'{side}-semantic-unsupported'] = [
                {'name': node['name'], 'line': node['line']}
                for node in nodes if node['semantic-status'] == 'unsupported-scope'
            ]
    except Exception as error:
        result['pre-code-weight'] = []
        result['post-code-weight'] = []
        result['error'] = str(error)
    return json.dumps(result) + '\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute triviality weights for YAML dataset")
    parser.add_argument('--input', required=True, help="Input YAML file")
    parser.add_argument('--output', required=True, help="Output JSONL file")
    parser.add_argument('--tokenizer', required=True, help="HuggingFace tokenizer path")
    parser.add_argument('--cache', required=True, help="Path to DF stats pickle file (loaded directly, or built from --data-dir then saved here)")
    parser.add_argument('--data-dir', default=None, help="Directory of .py files to build DF stats from (optional; if omitted, --cache must be a pre-built pkl)")
    parser.add_argument('--corpus-id', default=None, help='Declared name of the reference corpus when building a new DF cache')
    parser.add_argument('--workers', type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    if args.data_dir:
        calculate_df(args.data_dir, args.cache, args.corpus_id)
    elif not os.path.exists(args.cache):
        raise FileNotFoundError(f"Cache file not found: {args.cache}. Provide --data-dir to build it.")
    else:
        cache_data = load_corpus_cache(args.cache)
        if args.corpus_id and cache_data['corpus_metadata']['declared_source'] != args.corpus_id:
            raise ValueError(f"Declared corpus differs from DF cache: {args.corpus_id}")
    with open(args.cache, 'rb') as cache_file:
        cache_sha256 = hashlib.sha256(cache_file.read()).hexdigest()

    with open(args.input, 'r') as f:
        yaml_data = yaml.safe_load(f)

    items = list(yaml_data.items())
    results = [None] * len(items)
    chunksize = max(1, len(items) // max(args.workers * 8, 1))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, args.cache, cache_sha256),
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
