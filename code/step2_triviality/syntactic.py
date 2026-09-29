import ast
import math
from semantic import _capture_span, _pattern_captures


def extract_ast_nodes(code):
    """Extract AST tokens with type labels and (line, col_start, col_end, text)."""
    if not code.strip():
        return [], []
    tree = ast.parse(code)
    code_bytes = code.encode('utf-8')
    line_offsets = [0]
    for line in code.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line.encode('utf-8')))
    tokens = []
    token_types = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            for case in node.cases:
                for name, pattern_node in _pattern_captures(case.pattern):
                    line, start, end = _capture_span(code_bytes, line_offsets,
                                                     name, pattern_node)
                    tokens.append((line, start, end, name))
                    token_types.append('Name')
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                token_name = node.name
                if isinstance(node, ast.ClassDef):
                    name_start = node.col_offset + 6
                elif isinstance(node, ast.AsyncFunctionDef):
                    name_start = node.col_offset + 10
                else:
                    name_start = node.col_offset + 4
                name_end = name_start + len(token_name.encode('utf-8'))
                tokens.append((node.lineno, name_start, name_end, token_name))
                token_types.append(type(node).__name__)
        elif isinstance(node, ast.arg):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                tokens.append((node.lineno, node.col_offset,
                               node.col_offset + len(node.arg.encode('utf-8')), node.arg))
                token_types.append('arg')
        elif isinstance(node, ast.Name):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                tokens.append((node.lineno, node.col_offset, node.end_col_offset, node.id))
                token_types.append('Name')
        elif isinstance(node, ast.Attribute):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                attr_start = node.end_col_offset - len(node.attr.encode('utf-8'))
                tokens.append((node.end_lineno, attr_start, node.end_col_offset, node.attr))
                token_types.append('Attribute')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                target = node.func
                tokens.append((target.lineno, target.col_offset, target.end_col_offset, target.id))
                token_types.append('Call')
            elif isinstance(node.func, ast.Attribute):
                target = node.func
                start = target.end_col_offset - len(target.attr.encode('utf-8'))
                tokens.append((target.end_lineno, start, target.end_col_offset, target.attr))
                token_types.append('Call.Attribute')
        elif isinstance(node, ast.Constant):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                token_value = str(node.value)
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                start_col = node.col_offset
                end_col = getattr(node, 'end_col_offset', start_col + len(token_value))
                if start_line == end_line:
                    tokens.append((start_line, start_col, end_col, token_value))
                    token_types.append('Constant')
                else:
                    for line in range(start_line, end_line + 1):
                        if line == start_line:
                            tokens.append((line, start_col, 99999, token_value))
                        elif line == end_line:
                            tokens.append((line, 0, end_col, token_value))
                        else:
                            tokens.append((line, 0, 99999, token_value))
                        token_types.append('Constant')
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if hasattr(alias, 'lineno') and hasattr(alias, 'col_offset') and hasattr(alias, 'end_col_offset'):
                    name = alias.asname if alias.asname else alias.name
                    tokens.append((alias.lineno, alias.col_offset, alias.end_col_offset, name))
                    token_types.append('Import')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if hasattr(alias, 'lineno') and hasattr(alias, 'col_offset') and hasattr(alias, 'end_col_offset'):
                    name = alias.asname if alias.asname else alias.name
                    tokens.append((alias.lineno, alias.col_offset, alias.end_col_offset, name))
                    token_types.append('Import')

    combined = list(zip(tokens, token_types))
    combined.sort(key=lambda x: (x[0][0], x[0][1]))
    tokens = [x[0] for x in combined]
    token_types = [x[1] for x in combined]
    return tokens, token_types


def syntactic_score(code):
    """Paper Eq. (1): longest repeated suffix including the current AST node."""
    if not code.strip():
        return {}
    tokens, token_types = extract_ast_nodes(code)
    n = len(tokens)
    seq = [(token_types[i], tokens[i][3]) for i in range(n)]
    scores = {}
    previous = [0] * n
    for i in range(n):
        current = [0] * n
        for j in range(i):
            if seq[i] == seq[j]:
                current[j] = 1 + (previous[j - 1] if j else 0)
        length = max(current[:i], default=0)
        count = sum(value >= length for value in current[:i]) if length else 0
        score = (1 + count) ** (-math.log(length)) if count else 1.0
        scores[(*tokens[i], token_types[i])] = score
        previous = current

    return scores


def build_syntactic_line_index(syntactic_scores):
    """Group syntactic scores by line number for efficient span lookup."""
    line_index = {}
    for (line, start, end, token, node_type), score in syntactic_scores.items():
        line_index.setdefault(line, []).append((start, end, score, token, node_type))
    for line in line_index:
        line_index[line].sort(key=lambda x: x[0])
    return line_index


def syntactic_score_for_span(syntactic_line_index, token_line, token_col_start, token_col_end):
    """Return the max syntactic score overlapping a given column span on a line."""
    syntactic_weight = 0.0
    syntactic_source = []
    for syn_start, syn_end, score, syn_token, _ in syntactic_line_index.get(token_line, []):
        if syn_start >= token_col_end:
            break
        if syn_end <= token_col_start:
            continue
        overlap_start = max(token_col_start, syn_start)
        overlap_end = min(token_col_end, syn_end)
        if overlap_start < overlap_end and score > syntactic_weight:
            syntactic_weight = score
            syntactic_source = [syn_token]
    return syntactic_weight, syntactic_source
