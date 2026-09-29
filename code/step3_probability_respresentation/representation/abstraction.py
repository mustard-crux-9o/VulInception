"""Prefix-aware identifier abstraction for statement pairs."""

from __future__ import annotations

import ast
import builtins
import keyword
import sys
from collections import defaultdict, deque
from dataclasses import dataclass

from .normalization import normalize_statement


_PRESERVED_NAMES = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | set(getattr(sys, "stdlib_module_names", ()))
    | {"self", "cls"}
)


@dataclass(frozen=True)
class _Identifier:
    name: str
    category: str


class _IdentifierCollector(ast.NodeVisitor):
    """Collect identifiers in source order and assign coarse AST roles."""

    def __init__(self, preserved_names: set[str]) -> None:
        self.preserved_names = preserved_names
        self.identifiers: list[_Identifier] = []
        self._seen: set[str] = set()

    def _add(self, name: str, category: str) -> None:
        if name in self.preserved_names or name in self._seen:
            return
        self._seen.add(name)
        self.identifiers.append(_Identifier(name, category))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node.name, "FUNC")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node.name, "FUNC")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node.name, "CLASS")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._add(node.arg, "VAR")
        if node.annotation:
            self.visit(node.annotation)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._add(node.func.id, "FUNC")
        else:
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword_argument in node.keywords:
            self.visit(keyword_argument.value)

    def visit_Name(self, node: ast.Name) -> None:
        self._add(node.id, "VAR")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Attribute names are retained because they often denote library APIs.
        self.visit(node.value)

    def visit_alias(self, node: ast.alias) -> None:
        if node.asname:
            self._add(node.asname, "MODULE")


class _IdentifierRewriter(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self.mapping.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.mapping.get(node.id, node.id)
        return node

    def visit_alias(self, node: ast.alias) -> ast.alias:
        if node.asname:
            node.asname = self.mapping.get(node.asname, node.asname)
        return node


def _collect(code: str, preserved_names: set[str]) -> list[_Identifier]:
    tree = ast.parse(code, mode="exec")
    collector = _IdentifierCollector(preserved_names)
    collector.visit(tree)
    return collector.identifiers


def _rewrite(code: str, mapping: dict[str, str]) -> str:
    tree = ast.parse(code, mode="exec")
    tree = _IdentifierRewriter(mapping).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


class PrefixAwareAbstractor:
    """Maintain mappings from the restored original prefix.

    Each call normalizes and abstracts one original/generated statement pair.
    New generated identifiers are aligned by AST role and first occurrence with
    identifiers in the corresponding original statement. Generated aliases are
    discarded after the pair, while original mappings become prefix state for
    the next call.
    """

    def __init__(self, preserved_names: set[str] | None = None) -> None:
        self.preserved_names = set(_PRESERVED_NAMES)
        if preserved_names:
            self.preserved_names.update(preserved_names)
        self.mapping: dict[str, str] = {}
        self._counters: defaultdict[str, int] = defaultdict(int)

    def _allocate(self, category: str) -> str:
        index = self._counters[category]
        self._counters[category] += 1
        return f"{category}_{index}"

    def observe_original(self, statement: str) -> str:
        """Add an original prefix statement without a generated counterpart."""
        normalized = normalize_statement(statement)
        for identifier in _collect(normalized, self.preserved_names):
            if identifier.name not in self.mapping:
                self.mapping[identifier.name] = self._allocate(identifier.category)
        return _rewrite(normalized, self.mapping)

    def abstract_pair(self, original: str, generated: str) -> tuple[str, str]:
        """Normalize and abstract a statement pair under one base mapping."""
        original = normalize_statement(original)
        generated = normalize_statement(generated)

        original_identifiers = _collect(original, self.preserved_names)
        for identifier in original_identifiers:
            if identifier.name not in self.mapping:
                symbol = self._allocate(identifier.category)
                self.mapping[identifier.name] = symbol

        available: dict[str, deque[str]] = defaultdict(deque)
        for identifier in original_identifiers:
            available[identifier.category].append(self.mapping[identifier.name])

        pair_mapping = dict(self.mapping)
        for identifier in _collect(generated, self.preserved_names):
            if identifier.name in pair_mapping:
                symbol = pair_mapping[identifier.name]
                if symbol in available[identifier.category]:
                    available[identifier.category].remove(symbol)
                continue
            if available[identifier.category]:
                pair_mapping[identifier.name] = available[identifier.category].popleft()
            else:
                pair_mapping[identifier.name] = self._allocate(identifier.category)

        return _rewrite(original, pair_mapping), _rewrite(generated, pair_mapping)

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the persistent original-prefix mapping."""
        return dict(self.mapping)
