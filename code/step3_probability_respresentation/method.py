"""Core inference logic: triviality-aware split-then-generate with echo logprob."""
import math
import json
import threading
import concurrent.futures
from functools import lru_cache
from transformers import AutoTokenizer
from api import generate_text_batch_api, get_echo_logprob_batch_api

_thread_local = threading.local()


def load_weight_data(weight_file_path):
    """Load triviality weight JSONL into a dict keyed by CVE name."""
    weight_dict = {}
    with open(weight_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                cve_name = record.get('cve-name')
                if cve_name:
                    weight_dict[cve_name] = {
                        'pre': record.get('pre-code-weight', []),
                        'post': record.get('post-code-weight', [])
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


def build_line_weight_map(weight_list):
    line_weight_map = {}
    for item in weight_list:
        line = item.get('line')
        if line is None:
            continue
        line_weight_map.setdefault(line, []).append(item)
    return line_weight_map


def build_line_start_offsets(code):
    offsets = [0]
    for idx, ch in enumerate(code):
        if ch == '\n':
            offsets.append(idx + 1)
    return offsets


def get_line_weights(weight_index, line_num):
    if isinstance(weight_index, dict):
        return weight_index.get(line_num, [])
    return [w for w in weight_index if w.get('line') == line_num]


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


def prepare_line_requests(line_num, prefix_code, prefix_token_count,
                          current_line, current_encoding, offset_mapping, line_weights, tokenizer):
    current_ids = current_encoding['input_ids']
    if not current_ids:
        gen_meta = {'has_request': False, 'split_offset': [line_num, 0, ''], 'prefix_before_split': ''}
        echo_meta = {'has_echo': False, 'prefix_token_count': 0, 'split_pos': 0, 'token_weights': [], 'current_len': 0}
        return None, gen_meta, None, echo_meta

    split_pos, char_position, split_token_str = _find_indent_split_pos(
        current_line, current_ids, offset_mapping, tokenizer)
    prefix_before_split = current_line[:char_position]

    gen_meta = {'has_request': True, 'split_offset': [line_num, char_position, split_token_str],
                'prefix_before_split': prefix_before_split}

    if not line_weights:
        echo_meta = {'has_echo': False, 'prefix_token_count': 0, 'split_pos': 0, 'token_weights': [], 'current_len': 0}
        return prefix_code + prefix_before_split, gen_meta, None, echo_meta

    token_weights = []
    for j in range(split_pos, len(current_ids)):
        tw = line_weights[j]['weight'] if j < len(line_weights) else 1.0
        token_weights.append(tw)

    current_decoded = tokenizer.decode(current_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    echo_prompt = prefix_code + current_decoded
    echo_meta = {
        'has_echo': True, 'prefix_token_count': prefix_token_count,
        'split_pos': split_pos, 'token_weights': token_weights, 'current_len': len(current_ids)
    }
    return prefix_code + prefix_before_split, gen_meta, echo_prompt, echo_meta


def infer_record(record, config, weight_dict, tokenizer=None):
    """Run split-then-generate inference on one CVE record, returning probs + triviality."""
    pre_code = record.get('pre-code', '')
    post_code = record.get('post-code', '')
    if tokenizer is None and (pre_code or post_code):
        tokenizer = get_tokenizer(config['TARGET_MODEL_PATH'])

    cve_name = record.get('CVE-name', '')
    weight_data = weight_dict.get(cve_name, {'pre': [], 'post': []})
    pre_weight_map = build_line_weight_map(weight_data['pre'])
    post_weight_map = build_line_weight_map(weight_data['post'])

    result = {
        'pre-label': max(record.get('pre-label', 0), 0),
        'post-label': max(record.get('post-label', 0), 0),
        'pre-code': pre_code, 'post-code': post_code,
        'pre-criteria-lines': record.get('pre-criteria-lines', []),
        'pre-relative-lines': record.get('pre-relative-lines', []),
        'post-criteria-lines': record.get('post-criteria-lines', []),
        'post-relative-lines': record.get('post-relative-lines', []),
        'pre-split_offset': [], 'post-split_offset': [],
        'pre-probs': [], 'post-probs': [],
        'pre-triviality': [], 'post-triviality': [],
        'pre-target-PPL': [], 'post-target-PPL': [],
        'pre-ref-PPL': None, 'post-ref-PPL': None,
        'pre-ratio-PPL': None, 'post-ratio-PPL': None,
    }

    if pre_code:
        lines = record.get('pre-criteria-lines', []) + record.get('pre-relative-lines', [])
        gen, splits, probs, triv, ppls = _process_code_lines(pre_code, lines, tokenizer, config, pre_weight_map)
        result['generated-pre-code'] = '\n'.join(gen)
        result['pre-split_offset'] = splits
        result['pre-probs'] = probs
        result['pre-triviality'] = triv
        result['pre-target-PPL'] = ppls

    if post_code:
        lines = record.get('post-criteria-lines', []) + record.get('post-relative-lines', [])
        gen, splits, probs, triv, ppls = _process_code_lines(post_code, lines, tokenizer, config, post_weight_map)
        result['generated-post-code'] = '\n'.join(gen)
        result['post-split_offset'] = splits
        result['post-probs'] = probs
        result['post-triviality'] = triv
        result['post-target-PPL'] = ppls

    return result


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
            prefix_token_count = len(enc['input_ids'])
        else:
            prefix_code = ''
            prefix_token_count = 0
        current_line = code_lines[line_num - 1]
        if tok.is_fast:
            cur_enc = tok(current_line, add_special_tokens=False, truncation=True,
                          max_length=max_new_tokens, return_offsets_mapping=True)
            offset_mapping = cur_enc.get('offset_mapping')
        else:
            cur_enc = tok(current_line, add_special_tokens=False, truncation=True, max_length=max_new_tokens)
            offset_mapping = None
        line_weights = get_line_weights(weight_index, line_num)
        return prepare_line_requests(
            line_num, prefix_code, prefix_token_count,
            current_line, cur_enc, offset_mapping, line_weights, tok
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

    split_offsets, probs_out, triviality_out, ppls_out = [], [], [], []
    gen_idx = echo_idx = 0
    for i, line_num in enumerate(valid_lines):
        gm = gen_meta_list[i]
        em = echo_meta_list[i]
        if gm['has_request']:
            gen_text = gen_results[gen_idx] if gen_idx < len(gen_results) else ''
            generated_lines[line_num - 1] = gm['prefix_before_split'] + gen_text
            gen_idx += 1
        split_offsets.append(gm['split_offset'])

        probs_list, triviality_list, ppls_list = [], [], []
        if em['has_echo']:
            token_logprobs = echo_results[echo_idx] if echo_idx < len(echo_results) else None
            echo_idx += 1
            n = em['current_len'] - em['split_pos']
            for j in range(n):
                pos = em['prefix_token_count'] + em['split_pos'] + j
                lp = token_logprobs[pos] if token_logprobs and pos < len(token_logprobs) else None
                tw = em['token_weights'][j] if j < len(em['token_weights']) else 1.0
                if lp is not None:
                    probs_list.append(math.exp(float(lp)))
                    ppls_list.append(math.exp(-float(lp)))
                else:
                    probs_list.append(0.0)
                    ppls_list.append(0.0)
                triviality_list.append(tw)
        probs_out.append(probs_list)
        triviality_out.append(triviality_list)
        ppls_out.append(ppls_list)

    return generated_lines, split_offsets, probs_out, triviality_out, ppls_out

