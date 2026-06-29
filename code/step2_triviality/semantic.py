import ast
import math
from collections import defaultdict


def build_next_use_index(tokens, token_types):
    """Map each 'def' position to the index of the next 'use' of that name."""
    next_use_after = {}
    next_use_index = [None] * len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        token_name = tokens[i][3]
        token_type = token_types[i]
        if token_type == 'def':
            next_use_index[i] = next_use_after.get(token_name)
        if token_type == 'use':
            next_use_after[token_name] = i
    return next_use_index


def semantic_score(code):
    """Compute distance-weighted def/use semantic triviality per AST token."""
    if not code.strip():
        return {}
    tree = ast.parse(code)
    tokens = []
    token_types = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                token_name = node.name
                if isinstance(node, ast.ClassDef):
                    name_start = node.col_offset + 6
                elif isinstance(node, ast.AsyncFunctionDef):
                    name_start = node.col_offset + 10
                else:
                    name_start = node.col_offset + 4
                name_end = name_start + len(token_name)
                tokens.append((node.lineno, name_start, name_end, token_name))
                token_types.append('def')
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if hasattr(arg, 'lineno') and hasattr(arg, 'col_offset') and hasattr(arg, 'end_col_offset'):
                        tokens.append((arg.lineno, arg.col_offset, arg.end_col_offset, arg.arg))
                        token_types.append('def')
                for default in node.args.defaults:
                    if hasattr(default, 'lineno') and hasattr(default, 'col_offset') and hasattr(default, 'end_col_offset'):
                        if isinstance(default, ast.Constant):
                            tokens.append((default.lineno, default.col_offset, default.end_col_offset, str(default.value)))
                            token_types.append('use')
        elif isinstance(node, ast.Name):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                tokens.append((node.lineno, node.col_offset, node.end_col_offset, node.id))
                if isinstance(node.ctx, ast.Store):
                    token_types.append('def')
                else:
                    token_types.append('use')
        elif isinstance(node, ast.Attribute):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                attr_start = node.end_col_offset - len(node.attr)
                tokens.append((node.lineno, attr_start, node.end_col_offset, node.attr))
                token_types.append('use')
        elif isinstance(node, ast.Constant):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                token_value = str(node.value)
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                start_col = node.col_offset
                end_col = getattr(node, 'end_col_offset', start_col + len(token_value))
                if start_line == end_line:
                    tokens.append((start_line, start_col, end_col, token_value))
                    token_types.append('use')
                else:
                    for line in range(start_line, end_line + 1):
                        if line == start_line:
                            tokens.append((line, start_col, 99999, token_value))
                        elif line == end_line:
                            tokens.append((line, 0, end_col, token_value))
                        else:
                            tokens.append((line, 0, 99999, token_value))
                        token_types.append('use')
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if hasattr(alias, 'lineno') and hasattr(alias, 'col_offset') and hasattr(alias, 'end_col_offset'):
                    name = alias.asname if alias.asname else alias.name
                    tokens.append((alias.lineno, alias.col_offset, alias.end_col_offset, name))
                    token_types.append('def')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if hasattr(alias, 'lineno') and hasattr(alias, 'col_offset') and hasattr(alias, 'end_col_offset'):
                    name = alias.asname if alias.asname else alias.name
                    tokens.append((alias.lineno, alias.col_offset, alias.end_col_offset, name))
                    token_types.append('def')

    combined = list(zip(tokens, token_types))
    combined.sort(key=lambda x: (x[0][0], x[0][1]))
    tokens = [x[0] for x in combined]
    token_types = [x[1] for x in combined]
    next_use_index = build_next_use_index(tokens, token_types)

    scores = {}
    for idx, (line_v, start_v, end_v, token_v) in enumerate(tokens):
        weighted_usage = defaultdict(float)
        for i in range(idx):
            line_u, start_u, end_u, token_u = tokens[i]
            type_u = token_types[i]
            d_uv = abs(line_v - line_u)
            if d_uv == 0:
                d_uv = 1
            xi = 1
            if type_u == 'def':
                next_use = next_use_index[i]
                if next_use is None or next_use >= idx:
                    xi = 0.5
            weighted_usage[token_u] += math.exp(-xi * d_uv)
        if token_v not in weighted_usage:
            d_sem = 0.0
        else:
            denom = sum(weighted_usage.values()) + 1e-8
            d_sem = weighted_usage[token_v] / denom
        scores[(line_v, start_v, end_v, token_v)] = 1 - d_sem

    return scores


def build_semantic_line_index(semantic_scores):
    """Group semantic scores by line number for efficient span lookup."""
    line_index = defaultdict(list)
    for (line, start, end, token), score in semantic_scores.items():
        line_index[line].append((start, end, score, token))
    for line in line_index:
        line_index[line].sort(key=lambda x: x[0])
    return line_index


def semantic_score_for_span(semantic_line_index, token_line, token_col_start, token_col_end):
    """Return the max semantic score overlapping a given column span on a line."""
    semantic_weight = 0.0
    semantic_source = []
    for sem_start, sem_end, score, sem_token in semantic_line_index.get(token_line, []):
        if sem_start >= token_col_end:
            break
        if sem_end <= token_col_start:
            continue
        overlap_start = max(token_col_start, sem_start)
        overlap_end = min(token_col_end, sem_end)
        if overlap_start < overlap_end and score > semantic_weight:
            semantic_weight = score
            semantic_source = [sem_token]
    return semantic_weight, semantic_source
