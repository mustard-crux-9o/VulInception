import ast
import math


def extract_ast_nodes(code):
    """Extract AST tokens with type labels and (line, col_start, col_end, text)."""
    if not code.strip():
        return [], []
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
                token_types.append(type(node).__name__)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if hasattr(arg, 'lineno') and hasattr(arg, 'col_offset') and hasattr(arg, 'end_col_offset'):
                        tokens.append((arg.lineno, arg.col_offset, arg.end_col_offset, arg.arg))
                        token_types.append('arg')
                for default in node.args.defaults:
                    if hasattr(default, 'lineno') and hasattr(default, 'col_offset') and hasattr(default, 'end_col_offset'):
                        if isinstance(default, ast.Constant):
                            tokens.append((default.lineno, default.col_offset, default.end_col_offset, str(default.value)))
                            token_types.append('Constant')
        elif isinstance(node, ast.Name):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                tokens.append((node.lineno, node.col_offset, node.end_col_offset, node.id))
                token_types.append('Name')
        elif isinstance(node, ast.Attribute):
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
                attr_start = node.end_col_offset - len(node.attr)
                tokens.append((node.lineno, attr_start, node.end_col_offset, node.attr))
                token_types.append('Attribute')
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


def syntactic_score(code, alpha=0.5):
    """Compute repetition-based syntactic triviality: lower score = more trivial."""
    if not code.strip():
        return {}
    tokens, token_types = extract_ast_nodes(code)
    n = len(tokens)
    seq = [(token_types[i], tokens[i][3]) for i in range(n)]
    scores = {}

    for i in range(n):
        prefix_seq = seq[:i]
        target = seq[i]
        best_L = 0
        for L in range(1, i + 1):
            candidate = tuple(seq[i - L:i])
            found = False
            for j in range(i - L):
                if tuple(prefix_seq[j:j + L]) == candidate:
                    found = True
                    break
            if found:
                best_L = L
            else:
                break

        if best_L == 0:
            scores[(tokens[i][0], tokens[i][1], tokens[i][2], tokens[i][3])] = 1.0
            continue

        pre = tuple(seq[i - best_L:i])
        count_pre = 0
        count_pre_target = 0
        for j in range(best_L, i + 1):
            window = tuple(seq[j - best_L:j])
            if window == pre:
                count_pre += 1
                if j < len(seq) and seq[j] == target:
                    count_pre_target += 1

        P = count_pre_target / count_pre if count_pre > 0 else 0.0
        W = 1.0 - math.exp(-alpha * best_L)
        dsyn = P * W
        scores[(tokens[i][0], tokens[i][1], tokens[i][2], tokens[i][3])] = 1 - dsyn

    return scores


def build_syntactic_line_index(syntactic_scores):
    """Group syntactic scores by line number for efficient span lookup."""
    line_index = {}
    for (line, start, end, token), score in syntactic_scores.items():
        line_index.setdefault(line, []).append((start, end, score, token))
    for line in line_index:
        line_index[line].sort(key=lambda x: x[0])
    return line_index


def syntactic_score_for_span(syntactic_line_index, token_line, token_col_start, token_col_end):
    """Return the max syntactic score overlapping a given column span on a line."""
    syntactic_weight = 0.0
    syntactic_source = []
    for syn_start, syn_end, score, syn_token in syntactic_line_index.get(token_line, []):
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
