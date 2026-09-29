"""Source-ordered statement spans for the release feature pipeline."""
import ast
import io
import tokenize
import textwrap
import re
from asttokens import ASTTokens


def _offsets(code):
    starts = [0]
    for line in code.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _header_end(code, start):
    """End of a compound statement's first top-level colon."""
    snippet = code[start:]
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO(snippet).readline):
        if tok.type == tokenize.OP:
            if tok.string in '([{':
                depth += 1
            elif tok.string in ')]}':
                depth -= 1
            elif tok.string == ':' and depth == 0:
                line = snippet.splitlines(keepends=True)
                return start + sum(len(x) for x in line[:tok.end[0] - 1]) + tok.end[1]
    raise ValueError(f'No terminating colon in compound statement at offset {start}')


def extract_statements(code):
    """Return non-overlapping original statements, including compound headers."""
    atok = ASTTokens(code, parse=True)
    tree = atok.tree
    starts = _offsets(code)
    result = []

    def visit(node):
        if isinstance(node, ast.stmt):
            compound = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                         ast.If, ast.For, ast.AsyncFor, ast.While,
                                         ast.With, ast.AsyncWith, ast.Try,
                                         getattr(ast, 'TryStar', ast.Try), ast.Match))
            begin, full_end = atok.get_text_range(node)
            end = _header_end(code, begin) if compound else full_end
            if end > begin:
                text = textwrap.dedent(code[begin:end]).strip()
                result.append({'start': begin, 'end': end, 'start_line': node.lineno,
                               'end_line': code.count('\n', 0, end) + 1,
                               'text': text, 'kind': type(node).__name__})
            if compound:
                for _, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for child in value:
                            if isinstance(child, ast.stmt):
                                visit(child)
                            elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                                for part in child.body:
                                    visit(part)
    for node in tree.body:
        visit(node)
    for line_no, line in enumerate(code.splitlines(keepends=True), 1):
        stripped = line.strip()
        if re.match(r'^(else|finally|except(?:\*|\b)).*:\s*$', stripped):
            begin = starts[line_no - 1] + len(line) - len(line.lstrip())
            result.append({'start': begin, 'end': begin + len(stripped),
                           'start_line': line_no, 'end_line': line_no,
                           'text': stripped, 'kind': 'ControlHeader'})
    # Traversal through AST fields may encounter each body once; keep stable spans.
    return sorted({(s['start'], s['end']): s for s in result}.values(), key=lambda s: s['start'])


def prepare_snippet(text):
    """Make a control header parseable without passing its temporary body downstream."""
    text = textwrap.dedent(text).strip()
    if not text:
        raise ValueError('Empty statement')
    if text.rstrip().endswith(':'):
        if text.startswith('elif '):
            return 'if ' + text[5:] + '\n    pass', 'elif'
        if text.startswith('else:'):
            return 'if True:\n    pass\nelse:\n    pass', 'else'
        if text.startswith('except'):
            return 'try:\n    pass\n' + text + '\n    pass', 'except'
        if text.startswith('finally:'):
            return 'try:\n    pass\nfinally:\n    pass', 'finally'
        if text.startswith('try:'):
            return 'try:\n    pass\nexcept Exception:\n    pass', 'try'
        return text + '\n    pass', 'header'
    return text, None


def strip_scaffold(text, scaffolded):
    if scaffolded == 'else':
        return 'else:'
    if scaffolded == 'finally':
        return 'finally:'
    if scaffolded == 'except':
        return next((line for line in text.splitlines() if line.startswith('except')), 'except:')
    if scaffolded:
        header = text.splitlines()[0]
        return 'elif ' + header[3:] if scaffolded == 'elif' else header
    return text
