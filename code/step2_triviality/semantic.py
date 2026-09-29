"""CFG and binding-aware semantic non-triviality (paper Eq. 2-3)."""
import ast
import math
from bisect import bisect_right
from collections import defaultdict, deque


def _head_nodes(statement):
    stack = list(reversed(list(ast.iter_child_nodes(statement))))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.stmt):
            continue
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue  # independent lexical scopes
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _children(statement):
    result = []
    for _, value in ast.iter_fields(statement):
        if isinstance(value, list):
            for child in value:
                if isinstance(child, ast.stmt):
                    result.append(child)
                elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                    result.extend(child.body)
    return result


def _statements(function):
    stack = list(reversed(function.body))
    while stack:
        stmt = stack.pop()
        yield stmt
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(reversed(_children(stmt)))


class _CFG:
    def __init__(self):
        self.edges = defaultdict(set)
        self.cache = {}
        self.exit = object()

    def connect(self, source, target):
        if target is not None:
            self.edges[source].add(target)

    def sequence(self, statements, following=None, break_to=None, continue_to=None, handlers=()):
        next_node = following
        for stmt in reversed(statements):
            node = id(stmt)
            self.edges[node]
            if isinstance(stmt, ast.If):
                self.connect(node, self.sequence(stmt.body, next_node, break_to, continue_to, handlers))
                self.connect(node, self.sequence(stmt.orelse, next_node, break_to, continue_to, handlers))
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                after = self.sequence(stmt.orelse, next_node, break_to, continue_to, handlers)
                self.connect(node, self.sequence(stmt.body, node, next_node, node, handlers))
                self.connect(node, after)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                self.connect(node, self.sequence(stmt.body, next_node, break_to, continue_to, handlers))
            elif isinstance(stmt, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                final = self.sequence(stmt.finalbody, next_node, break_to, continue_to, handlers)
                else_entry = self.sequence(stmt.orelse, final, break_to, continue_to, handlers)
                catches = tuple(self.sequence(h.body, final, break_to, continue_to, handlers)
                                for h in stmt.handlers)
                self.connect(node, self.sequence(stmt.body, else_entry, break_to, continue_to,
                                                 catches or handlers))
                for target in catches or handlers:
                    self.connect(node, target)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case_node = id(case)
                    self.connect(node, case_node)
                    self.connect(case_node, self.sequence(case.body, next_node,
                                                          break_to, continue_to, handlers))
                self.connect(node, next_node)
            elif isinstance(stmt, ast.Break):
                if break_to is None:
                    raise ValueError('break outside loop')
                self.connect(node, break_to)
            elif isinstance(stmt, ast.Continue):
                if continue_to is None:
                    raise ValueError('continue outside loop')
                self.connect(node, continue_to)
            elif isinstance(stmt, ast.Raise):
                for target in handlers or (self.exit,):
                    self.connect(node, target)
            elif isinstance(stmt, ast.Return):
                self.connect(node, self.exit)
            else:
                self.connect(node, next_node)
            if not isinstance(stmt, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                for target in handlers:
                    self.connect(node, target)
            next_node = node
        return next_node

    def distance(self, source, target):
        if source == target:
            return 0
        key = source, target
        if key in self.cache:
            return self.cache[key]
        queue = deque([(source, 0)])
        seen = {source}
        while queue:
            node, dist = queue.popleft()
            for neighbor in self.edges[node]:
                if neighbor == target:
                    self.cache[key] = dist + 1
                    return dist + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, dist + 1))
        self.cache[key] = None
        return None


def _bindings(function, statements):
    local = set()
    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        local.update(arg.arg for arg in _arguments(function.args))
    globals_, nonlocals = set(), set()
    for stmt in statements:
        if isinstance(stmt, ast.Global):
            globals_.update(stmt.names)
        elif isinstance(stmt, ast.Nonlocal):
            nonlocals.update(stmt.names)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local.add(stmt.name)
        if isinstance(stmt, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
            local.update(handler.name for handler in stmt.handlers if handler.name)
        if isinstance(stmt, ast.Match):
            local.update(name for case in stmt.cases
                         for name, _ in _pattern_captures(case.pattern))
        for node in _head_nodes(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                local.add(node.id)
            elif isinstance(node, ast.alias):
                local.add(node.asname or node.name.split('.')[0])
        # Assignment expressions inside comprehensions bind in the enclosing
        # function or class, unlike the comprehension's for-targets.
        pending = [stmt]
        while pending:
            node = pending.pop()
            if node is not stmt and (isinstance(node, ast.stmt) or isinstance(node, ast.Lambda)):
                continue
            if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                local.add(node.target.id)
            pending.extend(ast.iter_child_nodes(node))
    return local - globals_ - nonlocals, globals_, nonlocals


def _arguments(args):
    values = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    values.extend(arg for arg in (args.vararg, args.kwarg) if arg is not None)
    return values


def _pattern_captures(pattern):
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            yield node.name, node
        elif isinstance(node, ast.MatchMapping) and node.rest:
            yield node.rest, node


def _capture_span(code_bytes, line_offsets, name, node):
    start = line_offsets[node.lineno - 1] + node.col_offset
    end = line_offsets[node.end_lineno - 1] + node.end_col_offset
    found = code_bytes.rfind(name.encode('utf-8'), start, end)
    if found < 0:
        raise ValueError(f'Cannot locate match capture {name!r} at line {node.lineno}')
    line = bisect_right(line_offsets, found)
    column = found - line_offsets[line - 1]
    return line, column, column + len(name.encode('utf-8'))


def _scoped_nodes(statement, binding):
    """Visit one statement's expressions with lambda/comprehension-local bindings."""
    expression_scopes = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def resolve(name, scopes):
        for scope_node, local in reversed(scopes):
            if name in local:
                return ('expression', id(scope_node), name)
        return binding(name, statement)

    def visit(node, scopes=()):
        if node is not statement and isinstance(node, ast.stmt):
            return
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *[x for x in node.args.kw_defaults if x is not None]):
                yield from visit(default, scopes)
            local = {arg.arg for arg in _arguments(node.args)}
            pending = [node.body]
            while pending:
                part = pending.pop()
                if isinstance(part, ast.Lambda):
                    continue
                if isinstance(part, ast.NamedExpr) and isinstance(part.target, ast.Name):
                    local.add(part.target.id)
                pending.extend(ast.iter_child_nodes(part))
            inner = (*scopes, (node, local))
            for arg in _arguments(node.args):
                yield arg, resolve(arg.arg, inner), lambda name, scopes=inner: resolve(name, scopes)
            yield from visit(node.body, inner)
            return
        if isinstance(node, expression_scopes[1:]):
            local = {name.id for gen in node.generators for name in ast.walk(gen.target)
                     if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}
            inner = (*scopes, (node, local))
            for index, generator in enumerate(node.generators):
                yield from visit(generator.iter, scopes if index == 0 else inner)
                yield from visit(generator.target, inner)
                for condition in generator.ifs:
                    yield from visit(condition, inner)
            if isinstance(node, ast.DictComp):
                yield from visit(node.key, inner)
                yield from visit(node.value, inner)
            else:
                yield from visit(node.elt, inner)
            return
        if isinstance(node, ast.Name):
            yield node, resolve(node.id, scopes), lambda name, scopes=scopes: resolve(name, scopes)
        elif isinstance(node, ast.alias):
            name = node.asname or node.name.split('.')[0]
            yield node, resolve(name, scopes), lambda name, scopes=scopes: resolve(name, scopes)
        else:
            for child in ast.iter_child_nodes(node):
                yield from visit(child, scopes)

    yield from visit(statement)


def _handler_bindings(statements):
    """Map statements inside an except body to its temporary alias binding."""
    scopes = defaultdict(dict)
    for stmt in statements:
        if not isinstance(stmt, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
            continue
        for handler in stmt.handlers:
            if not handler.name:
                continue
            pending = list(handler.body)
            while pending:
                child = pending.pop()
                scopes[id(child)][handler.name] = ('except', id(handler), handler.name)
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    pending.extend(_children(child))
    return scopes


def semantic_score(code):
    tree = ast.parse(code)
    scores = {}
    code_bytes = code.encode('utf-8')
    line_offsets = [0]
    for line in code.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line.encode('utf-8')))
    scopes = [tree, *(n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))]
    for function in scopes:
        cfg = _CFG()
        cfg.connect(id(function), cfg.sequence(function.body, following=cfg.exit))
        statements = list(_statements(function))
        local, globals_, nonlocals = _bindings(function, statements)
        handler_bindings = _handler_bindings(statements)
        events = []

        def binding(name, statement):
            if name in handler_bindings[id(statement)]:
                return handler_bindings[id(statement)][name]
            return ('global' if name in globals_ else
                    'nonlocal' if name in nonlocals else
                    'local' if name in local else 'external', name)

        args = _arguments(function.args) if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) else ()
        for arg in args:
            events.append((arg.lineno, arg.col_offset, arg.end_col_offset, arg.arg,
                           id(function), 'def', ('local', arg.arg),
                           lambda name: ('local', name)))
        for stmt in statements:
            lambda_parameters = {}
            guard_sources = {}
            for expression in ast.walk(stmt):
                if isinstance(expression, ast.Lambda):
                    cfg.connect(id(expression), id(stmt))
                    lambda_parameters.update((id(arg), id(expression))
                                             for arg in _arguments(expression.args))
            if isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    for name, pattern_node in _pattern_captures(case.pattern):
                        line, start, end = _capture_span(code_bytes, line_offsets,
                                                         name, pattern_node)
                        events.append((line, start, end, name, id(case), 'def',
                                       binding(name, stmt),
                                       lambda identifier, stmt=stmt: binding(identifier, stmt)))
                    if case.guard is not None:
                        guard_sources.update((id(node), id(case)) for node in ast.walk(case.guard)
                                             if isinstance(node, ast.Name))
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offset = 10 if isinstance(stmt, ast.AsyncFunctionDef) else 6 if isinstance(stmt, ast.ClassDef) else 4
                events.append((stmt.lineno, stmt.col_offset + offset,
                               stmt.col_offset + offset + len(stmt.name.encode('utf-8')), stmt.name,
                               id(stmt), 'def', binding(stmt.name, stmt),
                               lambda name, stmt=stmt: binding(name, stmt)))
            if isinstance(stmt, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
                for handler in stmt.handlers:
                    if handler.name:
                        events.append((handler.lineno, handler.col_offset, handler.col_offset,
                                       handler.name, id(handler.body[0]), 'def',
                                       ('except', id(handler), handler.name),
                                       lambda name, first=handler.body[0]: binding(name, first)))
            for node, node_binding, resolver in _scoped_nodes(stmt, binding):
                if isinstance(node, ast.Name):
                    events.append((node.lineno, node.col_offset, node.end_col_offset,
                                   node.id, guard_sources.get(id(node), id(stmt)),
                                   'def' if isinstance(node.ctx, ast.Store) else 'use',
                                   node_binding, resolver))
                elif isinstance(node, ast.alias):
                    name = node.asname or node.name.split('.')[0]
                    events.append((node.lineno, node.col_offset, node.end_col_offset,
                                   name, id(stmt), 'def', node_binding, resolver))
                elif isinstance(node, ast.arg):
                    events.append((node.lineno, node.col_offset,
                                   node.col_offset + len(node.arg.encode('utf-8')),
                                   node.arg, lambda_parameters.get(id(node), id(stmt)),
                                   'def', node_binding, resolver))
        events.sort(key=lambda e: (e[0], e[1], 0 if e[5] == 'use' else 1))
        last_use = {}
        for i, current in enumerate(events):
            if current[4] == id(function):
                continue
            usage = defaultdict(float)
            for j, prior in enumerate(events[:i]):
                if current[7](prior[3]) != prior[6]:
                    continue
                distance = cfg.distance(prior[4], current[4])
                if distance is None:
                    continue
                unused_definition = prior[5] == 'def' and last_use.get(prior[6], -1) <= j
                usage[prior[6]] += math.exp(-(0.5 if unused_definition else 1.0) * distance)
            value = 1.0 - usage[current[6]] / (sum(usage.values()) + 1e-8)
            scores[current[:4]] = max(0.0, min(1.0, value))
            if current[5] == 'use':
                last_use[current[6]] = i
    return scores


def build_semantic_line_index(semantic_scores):
    index = defaultdict(list)
    for (line, start, end, name), score in semantic_scores.items():
        index[line].append((start, end, score, name))
    for values in index.values():
        values.sort(key=lambda item: item[0])
    return index


def semantic_score_for_span(index, line, start, end):
    for left, right, score, name in index.get(line, ()):
        if left < end and start < right:
            return score, [name]
    return 1.0, []
