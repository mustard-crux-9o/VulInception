"""Deterministic Python statement normalization.

The implementation intentionally uses only Python's standard ``ast`` module.
It canonicalizes a small, auditable set of surface forms and leaves constructs
unchanged whenever the rewrite would have unclear runtime semantics.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterable


class NormalizationError(ValueError):
    """Raised when a snippet cannot be parsed as Python."""


def _load_context(node: ast.expr) -> ast.expr:
    """Return a copy of an assignment target that can be read."""
    result = copy.deepcopy(node)
    for child in ast.walk(result):
        if hasattr(child, "ctx"):
            child.ctx = ast.Load()
    return result


def _is_reorder_safe(node: ast.AST) -> bool:
    """Whether reordering the expression cannot reorder function calls."""
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        return _is_reorder_safe(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_reorder_safe(item) for item in node.elts)
    return False


def _stable_key(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class _Normalizer(ast.NodeTransformer):
    _COMMUTATIVE_OPERATORS = (
        ast.Add,
        ast.Mult,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
    )

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.Assign:
        """Canonicalize ``x += y`` as ``x = x + y``."""
        node = self.generic_visit(node)
        replacement = ast.Assign(
            targets=[node.target],
            value=ast.BinOp(
                left=_load_context(node.target),
                op=node.op,
                right=node.value,
            ),
        )
        return ast.copy_location(replacement, node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """Order side-effect-free operands of commutative operators."""
        node = self.generic_visit(node)
        if (
            isinstance(node.op, self._COMMUTATIVE_OPERATORS)
            and _is_reorder_safe(node.left)
            and _is_reorder_safe(node.right)
            and _stable_key(node.right) < _stable_key(node.left)
        ):
            node.left, node.right = node.right, node.left
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        """Canonicalize symmetric equality operands when both are safe."""
        node = self.generic_visit(node)
        if (
            len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and _is_reorder_safe(node.left)
            and _is_reorder_safe(node.comparators[0])
            and _stable_key(node.comparators[0]) < _stable_key(node.left)
        ):
            node.left, node.comparators[0] = node.comparators[0], node.left
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Canonicalize empty built-in constructors to literals."""
        node = self.generic_visit(node)
        if node.args or node.keywords or not isinstance(node.func, ast.Name):
            return node

        replacements: dict[str, ast.expr] = {
            "dict": ast.Dict(keys=[], values=[]),
            "list": ast.List(elts=[], ctx=ast.Load()),
            "tuple": ast.Tuple(elts=[], ctx=ast.Load()),
            "str": ast.Constant(value=""),
            "bytes": ast.Constant(value=b""),
        }
        replacement = replacements.get(node.func.id)
        return ast.copy_location(replacement, node) if replacement else node


def normalize_statement(code: str) -> str:
    """Normalize one or more Python statements into a canonical form."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise NormalizationError(str(error)) from error

    tree = _Normalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


def normalize_statements(statements: Iterable[str]) -> list[str]:
    """Normalize an ordered statement sequence."""
    return [normalize_statement(statement) for statement in statements]
