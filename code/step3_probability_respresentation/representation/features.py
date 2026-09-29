"""Finish target-model features before the classifier stage."""
import copy
import textwrap

import numpy as np

from . import PrefixAwareAbstractor
from preprocessing.statements import prepare_snippet, strip_scaffold


RICH_STATS_DIM = 22
SELECTION_MODES = ('all', 'criteria', 'relative', 'criteria+relative')


def compute_rich_stats(values):
    """The release's 22 statistics, evaluated on one statement's p(t) * tau(t)."""
    if not values:
        return [0.0] * RICH_STATS_DIM
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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


def _statement_stats(record, side, statement, line_starts, locations, selected_lines):
    weighted = []
    for line in range(statement['start_line'], statement['end_line'] + 1):
        if line not in selected_lines or line not in locations:
            continue
        index = locations[line]
        probs = record[f'{side}-probs'][index]
        weights = record[f'{side}-triviality'][index]
        spans = record[f'{side}-prob-spans'][index]
        if not (len(probs) == len(weights) == len(spans)):
            raise ValueError(f"{record.get('cve_name')}:{side}:{line}: probability/token weights are misaligned")
        for probability, weight, (left, right) in zip(probs, weights, spans):
            midpoint = line_starts[line - 1] + (left + right) / 2
            if statement['start'] <= midpoint < statement['end']:
                weighted.append(float(probability) * float(weight))
    return compute_rich_stats(weighted) if weighted else None


def _prepare_side(record, side):
    code = record[f'{side}-code']
    criteria = set(record[f'{side}-criteria-lines'])
    relative = set(record[f'{side}-relative-lines'])
    locations = {entry[0]: index for index, entry in enumerate(record[f'{side}-split_offset'])}
    selected = {'all': set(locations), 'criteria': criteria, 'relative': relative,
                'criteria+relative': criteria | relative}
    line_starts = [0]
    for line in code.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    generated_lines = (record.get(f'generated-{side}-code') or '').split('\n')
    abstractor = PrefixAwareAbstractor()
    prefix, results = [], []
    for statement in record[f'{side}-statements']:
        try:
            original, original_mode = prepare_snippet(statement['text'])
        except Exception as error:
            raise ValueError(f"{record.get('cve_name')}:{side}:{statement['start_line']}: original statement cannot be parsed") from error
        active = any(line in selected['criteria+relative']
                     for line in range(statement['start_line'], statement['end_line'] + 1))
        if active:
            generated = '\n'.join(generated_lines[statement['start_line'] - 1:statement['end_line']])
            generated = textwrap.dedent(generated).strip()
            generated_text = None
            status = 'ok'
            try:
                generated, generated_mode = prepare_snippet(generated)
                temporary = copy.deepcopy(abstractor)
                original_text, generated_text = temporary.abstract_pair(original, generated)
                original_text = strip_scaffold(original_text, original_mode)
                generated_text = strip_scaffold(generated_text, generated_mode)
            except Exception as error:
                status = str(error)
            observed = strip_scaffold(abstractor.observe_original(original), original_mode)
            if generated_text is None:
                original_text = observed
            features = {mode: _statement_stats(record, side, statement, line_starts, locations, lines)
                        for mode, lines in selected.items()}
            results.append({
                'start_line': statement['start_line'], 'end_line': statement['end_line'],
                'weight': float(statement['weight']), 'prob-stats': features,
                'abstracted-prefix': '\n'.join(prefix),
                'abstracted-original': original_text,
                'abstracted-generated': generated_text,
                'generated-status': status,
            })
            prefix.append(observed)
        else:
            prefix.append(strip_scaffold(abstractor.observe_original(original), original_mode))
    return results


def prepare_classifier_features(record):
    """Return only the data needed for prefix embedding and classifier training."""
    result = {
        'feature-version': 5,
        'corpus-metadata': record['corpus-metadata'],
        'pre-label': record['pre-label'], 'post-label': record['post-label'],
    }
    for side in ('pre', 'post'):
        result[f'{side}-has-code'] = bool(record[f'{side}-code'].strip())
        result[f'{side}-criteria-lines'] = record[f'{side}-criteria-lines']
        result[f'{side}-relative-lines'] = record[f'{side}-relative-lines']
        result[f'{side}-processed-lines'] = [entry[0] for entry in record[f'{side}-split_offset']]
        result[f'{side}-statements'] = _prepare_side(record, side)
    return result
