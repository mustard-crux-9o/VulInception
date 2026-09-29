"""Core inference logic: triviality-aware split-then-generate with echo logprob."""
import math
import json
import threading
import concurrent.futures
import sys
from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer
from api import generate_text_batch_api, get_echo_logprob_batch_api

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation.features import prepare_classifier_features

_thread_local = threading.local()


def load_weight_data(weight_file_path):
    """Load triviality weight JSONL into a dict keyed by CVE name."""
    weight_dict = {}
    with open(weight_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get('feature-version') != 3 or record.get('error'):
                    raise ValueError(f"Regenerate step2 weights for {record.get('cve-name')}: {record.get('error', 'old feature version')}")
                metadata = record.get('corpus-metadata')
                if (not isinstance(metadata, dict) or metadata.get('format_version') != 3
                        or not metadata.get('manifest_sha256') or not metadata.get('cache_sha256')):
                    raise ValueError(f"Regenerate step2 weights for {record.get('cve-name')}: missing corpus provenance")
                required = [f'{side}-{field}' for side in ('pre', 'post')
                            for field in ('code-weight', 'node-scores', 'statements', 'semantic-unsupported')]
                missing = [field for field in required if not isinstance(record.get(field), list)]
                if missing:
                    raise ValueError(f"Regenerate step2 weights for {record.get('cve-name')}: missing {', '.join(missing)}")
                cve_name = record.get('cve-name')
                if cve_name:
                    weight_dict[cve_name] = {
                        'pre': record.get('pre-code-weight', []),
                        'post': record.get('post-code-weight', []),
                        'pre-nodes': record.get('pre-node-scores', []),
                        'post-nodes': record.get('post-node-scores', []),
                        'pre-statements': record.get('pre-statements', []),
                        'post-statements': record.get('post-statements', []),
                        'pre-unsupported': record.get('pre-semantic-unsupported', []),
                        'post-unsupported': record.get('post-semantic-unsupported', []),
                        'corpus-metadata': record['corpus-metadata']
                    }
    return weight_dict


@lru_cache(maxsize=2)
def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.truncation_side = 'left'
    return tokenizer


def _get_thread_tokenizer(model_path):
    tokenizer = getattr(_thread_local, 'tokenizer', None)
    if tokenizer is None or getattr(_thread_local, 'model_path', None) != model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.truncation_side = 'left'
        _thread_local.tokenizer = tokenizer
        _thread_local.model_path = model_path
    return tokenizer


def build_line_start_offsets(code):
    offsets = [0]
    for idx, ch in enumerate(code):
        if ch == '\n':
            offsets.append(idx + 1)
    return offsets


def _find_indent_split_pos(current_line, current_ids, offset_mapping, tokenizer):
    stripped = current_line.lstrip()
    if not stripped:
        return len(current_ids), len(current_line), ''
    indent_end = len(current_line) - len(stripped)
    if offset_mapping and len(offset_mapping) == len(current_ids):
        for idx, (start, end) in enumerate(offset_mapping):
            if end > indent_end:
                return idx, indent_end, current_line[max(start, indent_end):end]
        return len(current_ids), len(current_line), ''
    char_pos = 0
    for idx, tid in enumerate(current_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        next_pos = char_pos + len(tok_text)
        if next_pos > indent_end:
            return idx, indent_end, tok_text.lstrip() or tok_text
        char_pos = next_pos
    return len(current_ids), len(current_line), ''


def prepare_line_requests(line_num, line_start, prefix_code, current_line,
                          current_encoding, offset_mapping, node_scores, tokenizer):
    current_ids = current_encoding['input_ids']
    if not current_ids:
        gen_meta = {'has_request': False, 'split_offset': [line_num, 0, ''], 'prefix_before_split': ''}
        echo_meta = {'has_echo': False, 'token_weights': [], 'token_spans': [], 'token_indices': []}
        return None, gen_meta, None, echo_meta

    _, char_position, split_token_str = _find_indent_split_pos(
        current_line, current_ids, offset_mapping, tokenizer)
    prefix_before_split = current_line[:char_position]

    gen_meta = {'has_request': True, 'split_offset': [line_num, char_position, split_token_str],
                'prefix_before_split': prefix_before_split}

    if offset_mapping is None:
        raise ValueError('A fast tokenizer with offsets is required to align p(t) and tau(t)')
    current_decoded = tokenizer.decode(current_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not current_line.startswith(current_decoded):
        raise ValueError(f'Line {line_num}: tokenized text is not a prefix of the source line')
    echo_prompt = prefix_code + current_decoded
    full = tokenizer(echo_prompt, add_special_tokens=False, return_offsets_mapping=True)
    body_start = len(prefix_code) + char_position
    token_indices, token_weights, token_spans = [], [], []
    for index, (start, end) in enumerate(full['offset_mapping']):
        if end <= body_start or start >= len(echo_prompt) or end <= start:
            continue
        local_start = max(0, start - len(prefix_code))
        local_end = min(len(current_decoded), end - len(prefix_code))
        abs_start = line_start + local_start
        abs_end = line_start + local_end
        matches = [node for node in node_scores
                   if node['abs_start'] < abs_end and abs_start < node['abs_end']]
        best = max(matches, key=lambda node: (
            min(abs_end, node['abs_end']) - max(abs_start, node['abs_start']),
            -(node['abs_end'] - node['abs_start']),
            node['type'] in ('Call', 'Call.Attribute'))) if matches else None
        token_indices.append(index)
        token_weights.append(float(best['weight']) if best else 0.0)
        token_spans.append((local_start, local_end))
    echo_meta = {
        'has_echo': bool(token_indices), 'prompt_ids': full['input_ids'],
        'token_indices': token_indices, 'token_weights': token_weights,
        'token_spans': token_spans,
    }
    return prefix_code + prefix_before_split, gen_meta, echo_prompt if token_indices else None, echo_meta


def _aligned_echo_logprobs(response, prompt_ids, line_num):
    """Verify the server's echo token IDs before indexing its log probabilities."""
    if not response or not isinstance(response.get('tokens'), list):
        raise ValueError(f'Line {line_num}: echo response has no token IDs')
    token_strings = response['tokens']
    try:
        server_ids = [int(token[9:]) if token.startswith('token_id:') else None
                      for token in token_strings]
    except (TypeError, ValueError) as error:
        raise ValueError(f'Line {line_num}: malformed echo token IDs') from error
    if None in server_ids:
        raise ValueError(f'Line {line_num}: serve vLLM with --return-tokens-as-token-ids')
    logprobs = response.get('token_logprobs')
    if not isinstance(logprobs, list) or len(logprobs) != len(server_ids):
        raise ValueError(f'Line {line_num}: echo tokens and log probabilities have different lengths')
    for shift in (0, 1):
        if server_ids[shift:shift + len(prompt_ids)] == prompt_ids:
            return logprobs, shift
    raise ValueError(f'Line {line_num}: local full-prompt tokens differ from vLLM echo tokens')


def infer_record(record, config, weight_dict, tokenizer=None):
    """Run target-model inference and return classifier-ready statement features."""
    pre_code = record.get('pre-code', '')
    post_code = record.get('post-code', '')
    if tokenizer is None and (pre_code or post_code):
        tokenizer = get_tokenizer(config['TARGET_MODEL_PATH'])

    cve_name = record.get('CVE-name', '')
    if cve_name not in weight_dict:
        raise ValueError(f'Missing step2 weights for {cve_name}')
    weight_data = weight_dict[cve_name]
    result = {
        'cve_name': cve_name,
        'feature-version': 5,
        'corpus-metadata': weight_data['corpus-metadata'],
        'pre-label': max(record.get('pre-label', 0), 0),
        'post-label': max(record.get('post-label', 0), 0),
        'pre-code': pre_code, 'post-code': post_code,
        'pre-criteria-lines': record.get('pre-criteria-lines', []),
        'pre-relative-lines': record.get('pre-relative-lines', []),
        'post-criteria-lines': record.get('post-criteria-lines', []),
        'post-relative-lines': record.get('post-relative-lines', []),
        'pre-split_offset': [], 'post-split_offset': [],
        'pre-probs': [], 'post-probs': [],
        'pre-prob-spans': [], 'post-prob-spans': [],
        'pre-triviality': [], 'post-triviality': [],
        'pre-statements': weight_data['pre-statements'],
        'post-statements': weight_data['post-statements'],
    }

    if pre_code:
        lines = record.get('pre-criteria-lines', []) + record.get('pre-relative-lines', [])
        gen, splits, probs, triv, spans = _process_code_lines(pre_code, lines, tokenizer, config, weight_data['pre-nodes'])
        result['generated-pre-code'] = '\n'.join(gen)
        result['pre-split_offset'] = splits
        result['pre-probs'] = probs
        result['pre-prob-spans'] = spans
        result['pre-triviality'] = triv

    if post_code:
        lines = record.get('post-criteria-lines', []) + record.get('post-relative-lines', [])
        gen, splits, probs, triv, spans = _process_code_lines(post_code, lines, tokenizer, config, weight_data['post-nodes'])
        result['generated-post-code'] = '\n'.join(gen)
        result['post-split_offset'] = splits
        result['post-probs'] = probs
        result['post-prob-spans'] = spans
        result['post-triviality'] = triv

    return prepare_classifier_features(result)


def _process_code_lines(code, lines_to_process, tokenizer, config, weight_index):
    code_lines = code.split('\n')
    generated_lines = code_lines.copy()
    line_count = len(code_lines)
    if not lines_to_process or line_count == 0:
        return generated_lines, [], [], [], []
    valid_lines = sorted(set(ln for ln in lines_to_process if 2 < ln <= line_count))
    if not valid_lines:
        return generated_lines, [], [], [], []

    max_new_tokens = config['MAX_NEW_TOKENS_GENERATION']
    max_prefix_len = config['MAX_CONTEXT_LEN'] - max_new_tokens - 10
    ft_url = f"http://127.0.0.1:{config['FT_PORT']}"
    line_start_offsets = build_line_start_offsets(code)
    timeout = config.get('TIMEOUT_SECONDS', 150)
    if tokenizer is None:
        tokenizer = get_tokenizer(config['TARGET_MODEL_PATH'])
    max_workers = min(len(valid_lines), max(1, int(config.get('MAX_WORKERS', 1))))

    def prepare_single_line(line_num):
        tok = _get_thread_tokenizer(config['TARGET_MODEL_PATH']) if max_workers > 1 else tokenizer
        raw_prefix = code[:line_start_offsets[line_num - 1]]
        if raw_prefix:
            enc = tok(raw_prefix, add_special_tokens=False, truncation=True, max_length=max_prefix_len)
            prefix_code = tok.decode(enc['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            prefix_code = ''
        current_line = code_lines[line_num - 1]
        if tok.is_fast:
            cur_enc = tok(current_line, add_special_tokens=False, return_offsets_mapping=True)
            cur_enc['input_ids'] = cur_enc['input_ids'][:max_new_tokens]
            cur_enc['offset_mapping'] = cur_enc['offset_mapping'][:max_new_tokens]
            offset_mapping = cur_enc['offset_mapping']
        else:
            cur_enc = tok(current_line, add_special_tokens=False)
            cur_enc['input_ids'] = cur_enc['input_ids'][:max_new_tokens]
            offset_mapping = None
        line_start = line_start_offsets[line_num - 1]
        line_nodes = [node for node in weight_index
                      if node['abs_start'] < line_start + len(current_line)
                      and line_start < node['abs_end']]
        return prepare_line_requests(
            line_num, line_start, prefix_code,
            current_line, cur_enc, offset_mapping, line_nodes, tok
        )

    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            line_results = list(executor.map(prepare_single_line, valid_lines))
    else:
        line_results = [prepare_single_line(ln) for ln in valid_lines]

    gen_requests, gen_meta_list = [], []
    echo_prompts, echo_meta_list = [], []
    for gen_prefix, gen_meta, echo_prompt, echo_meta in line_results:
        if gen_prefix is not None:
            gen_requests.append(gen_prefix)
        gen_meta_list.append(gen_meta)
        if echo_meta['has_echo']:
            echo_prompts.append(echo_prompt)
        echo_meta_list.append(echo_meta)

    temperature = config.get('TEMP', 0.0)
    gen_results = generate_text_batch_api(ft_url, gen_requests, max_new_tokens,
                                          temperature=temperature, timeout=timeout) if gen_requests else []
    echo_results = get_echo_logprob_batch_api(ft_url, echo_prompts, timeout) if echo_prompts else []

    split_offsets, probs_out, triviality_out, spans_out = [], [], [], []
    gen_idx = echo_idx = 0
    for i, line_num in enumerate(valid_lines):
        gm = gen_meta_list[i]
        em = echo_meta_list[i]
        if gm['has_request']:
            gen_text = gen_results[gen_idx] if gen_idx < len(gen_results) else ''
            generated_lines[line_num - 1] = gm['prefix_before_split'] + gen_text
            gen_idx += 1
        split_offsets.append(gm['split_offset'])

        probs_list, triviality_list = [], []
        if em['has_echo']:
            response = echo_results[echo_idx] if echo_idx < len(echo_results) else None
            echo_idx += 1
            token_logprobs, shift = _aligned_echo_logprobs(response, em['prompt_ids'], line_num)
            for j, token_index in enumerate(em['token_indices']):
                lp = token_logprobs[shift + token_index]
                tw = em['token_weights'][j]
                if lp is not None:
                    probs_list.append(math.exp(float(lp)))
                else:
                    probs_list.append(0.0)
                triviality_list.append(tw)
        probs_out.append(probs_list)
        triviality_out.append(triviality_list)
        spans_out.append(em['token_spans'])

    return generated_lines, split_offsets, probs_out, triviality_out, spans_out
