#!/usr/bin/env python3
"""Static leakage-firewall check -- Freeze-1 regression gate 9 (protocol section 5.2, item 9).

Gold answers, qrels, evidence sets and negative document sets are evaluator-only material. If any
of them can reach a selector, aggregator, preflight, broker feature or predictor feature, then the
treatment is being scored on its ability to read its own answer key, and every number downstream of
that -- contract risk, headroom, the joint-vs-sequential comparison -- is void rather than merely
noisy. The damage is invisible in the results: a leaked run looks like a very good run.

So the firewall is enforced structurally. This check walks the AST of every module on the treatment
path and rejects any way of naming evaluator material:

* a direct import of an evaluator-only module;
* a *transitive* one -- treatment imports X, X imports qrels. X is not innocent: importing X
  executes qrels' module body and puts it one attribute access away;
* a *function-local* one. A lazy import inside a function is the obvious way this rule gets broken,
  because it looks local and temporary and never appears in the file's import block;
* a *dynamic* one -- ``importlib.import_module("shapeflow.bench.bcplus.qrels")`` has no Import node
  at all, so module names appearing as string literals are treated as import edges too;
* a reference to the evaluator-only data file (``browsecomp_plus_decrypted.jsonl``), or to an
  answer-key-shaped attribute (``gold_docs``, ``evidence_docs``, ``negative_docs``, ``qrel*``,
  ``answer_key``), which is how the material arrives when someone passes it in as a plain dict
  instead of importing it;
* a module inside the treatment path whose own dotted name is answer-key-shaped. The other rules
  all look inside a file, so a module *called* ``gold_docs`` with tidy internal names would
  otherwise be the one place the material can sit in a treatment package unremarked.

Scope is the treatment path *and its import closure*: a helper module that the treatment path
imports is part of the treatment path, whatever directory it lives in.

Two deliberate design choices:

* There is no exemption list. An exemption list is how a firewall dies -- the first entry is always
  justified and the tenth is never reviewed. Prose may name the material (the name rules ignore any
  string containing whitespace, so docstrings and messages are free); code may not.
* This file imports nothing from ``shapeflow``. A check that needs the package to be importable goes
  dark exactly when the tree is broken, which is exactly when someone is most likely to reach for
  the answer key to get moving again.

The check is honest about its limit: it is static, so a name assembled at runtime from fragments can
evade it. It is a firewall against the ways this actually gets broken, not a proof.

Usage: ``python tools/ci/check_leakage_firewall.py [SRC_ROOT]`` -- exit 0 clean, 1 on violation.
Wired into ``scripts/verify_local.sh``.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: A module declares itself evaluator-only with ``EVALUATOR_ONLY = True`` at module level
#: (``shapeflow.bench.bcplus.qrels`` does exactly this and says so in its docstring). The marker
#: exists so a new evaluator module joins the firewall by saying so in its own file, rather than by
#: someone remembering to edit the list below -- the edit that never happens is the leak. The
#: namespaced spelling is accepted too, because a module that has to be explicit about which
#: project's firewall it belongs to should not be silently unguarded for using a longer name.
MARKERS = ("EVALUATOR_ONLY", "SHAPEFLOW_EVALUATOR_ONLY")


@dataclass(frozen=True)
class FirewallSpec:
    """Who may not be reached, by whom, and under which names."""

    #: Packages that run on the treatment path. Their import closure is scanned too.
    treatment_packages: tuple[str, ...]
    #: Evaluator-only module names. These are banned as *names*, whether or not the file exists
    #: yet: a module still under construction must not become reachable the day it lands, and an
    #: import of a name that does not resolve is a leak attempt regardless of whether it works.
    evaluator_modules: tuple[str, ...]
    #: Substrings that make an identifier or a whitespace-free literal answer-key-shaped.
    forbidden_names: tuple[str, ...]
    #: Substrings that name evaluator-only data files.
    forbidden_files: tuple[str, ...]


#: The frozen Freeze-1 configuration. `shapeflow.bench.grading` scores an answer against the
#: benchmark's own ground truth and `shapeflow.bench.bcplus.qrels` holds the relevance labels;
#: `splits` and `corpus` are treatment-visible by design and are deliberately absent here.
FREEZE_1 = FirewallSpec(
    treatment_packages=(
        "shapeflow.broker",
        "shapeflow.evidence",
        "shapeflow.odr",
        "shapeflow.p1",
        "shapeflow.strategies",
    ),
    evaluator_modules=(
        "shapeflow.bench.bcplus.qrels",
        # Each of these also declares EVALUATOR_ONLY in its own file, which is the mechanism that
        # is supposed to make this list unnecessary. Named here anyway, because the marker rule
        # and the name rule fail differently: the marker is read from the module's source, so a
        # module that is deleted, renamed, or has its marker dropped in a refactor stops being
        # evaluator-only silently, and this list is what still refuses the import.
        "shapeflow.bench.bcplus.recall",
        "shapeflow.bench.bcplus.analysis",
        "shapeflow.bench.bcplus.render",
        # It holds the grading prompt and scores a prediction against the gold answer, and it was
        # the one marked module missing from this list -- so it had the marker rule protecting it
        # and not the name rule, which is exactly the asymmetry the comment above warns about.
        # ``test_every_marked_module_is_also_named`` now makes the omission impossible to repeat.
        "shapeflow.bench.bcplus.grader",
        # Freeze-2's judge-free quality family. It joins treatment-side trial records to the
        # benchmark's evidence and hard-negative sets, so it holds the answer key exactly as
        # much as recall does.
        "shapeflow.bench.bcplus.selection_quality",
        "shapeflow.bench.grading",
    ),
    forbidden_names=(
        "answer_key",
        # The per-query *evidence document set* is evaluator-only material (AGENTS.md section 2
        # names it beside gold and the negatives, and it is a field of the decrypted benchmark
        # file). The token is `evidence_doc` and not `evidence` because `shapeflow.evidence` is a
        # treatment package: banning the bare word would flag the compression path's own name and
        # the check would be turned off within a day. Omitting it entirely, which is how this
        # started, left `record["evidence_docs"]` -- a field read straight off the answer key --
        # passing the firewall while the failure message claimed evidence sets were covered.
        "evidence_doc",
        "gold_answer",
        "gold_doc",
        "negative_doc",
        "qrel",
    ),
    forbidden_files=("browsecomp_plus_decrypted",),
)


@dataclass(frozen=True)
class Violation:
    """One reason the firewall does not hold, with the line a person has to go and look at."""

    path: Path
    lineno: int
    rule: str
    detail: str

    def render(self, root: Path | None = None) -> str:
        shown: Path | str = self.path
        if root is not None:
            try:
                shown = self.path.relative_to(root)
            except ValueError:
                shown = self.path
        return f"{shown}:{self.lineno}: [{self.rule}] {self.detail}"


@dataclass(frozen=True)
class Report:
    violations: tuple[Violation, ...]
    treatment_modules: tuple[str, ...]
    closure_modules: tuple[str, ...]
    evaluator_modules: tuple[str, ...]
    evaluator_on_disk: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def rules(self) -> tuple[str, ...]:
        return tuple(v.rule for v in self.violations)


@dataclass(frozen=True)
class _Module:
    name: str
    path: Path
    is_package: bool
    tree: ast.Module


@dataclass(frozen=True)
class _Edge:
    """One way a module can cause another module's body to execute."""

    source: str
    target: str
    lineno: int
    #: "import" | "function-local import" | "dynamic module-name reference"
    kind: str


# --------------------------------------------------------------------------------------------
# module discovery
# --------------------------------------------------------------------------------------------


def _module_name(src_root: Path, path: Path) -> str | None:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _collect(src_root: Path) -> tuple[dict[str, _Module], list[Violation]]:
    """Parse every module under ``src_root``.

    Glob order is filesystem order, which differs between machines, so the sort is what makes two
    runs of this check produce byte-identical output. An unparseable file is a hard failure and not
    a skip: a module the check cannot read is a module the check cannot clear.
    """
    modules: dict[str, _Module] = {}
    violations: list[Violation] = []
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = _module_name(src_root, path)
        if name is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            violations.append(Violation(
                path, getattr(exc, "lineno", 0) or 0, "unparseable",
                f"cannot be parsed ({exc.__class__.__name__}), so the firewall cannot certify it: "
                "a module the check cannot read is not a module the check has cleared"))
            continue
        modules[name] = _Module(name, path, path.name == "__init__.py", tree)
    return modules, violations


def _marker_names(node: ast.AST) -> list[str]:
    """Marker names bound by one assignment target, including inside a tuple or list unpack.

    ``EVALUATOR_ONLY, VERSION = True, 1`` binds the marker just as surely as a plain assignment
    does. Looking only at bare ``Name`` targets read that statement as "no marker here", which is
    the fail-open direction: the module believes it declared itself evaluator-only and the firewall
    treats it as ordinary treatment-visible code.
    """
    return sorted({
        child.id for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in MARKERS})


def _marked_evaluator(modules: dict[str, _Module]) -> tuple[set[str], list[Violation]]:
    """Find modules that declare themselves evaluator-only, at module level only.

    A marker inside a function, a class body, an ``if`` or a ``try`` is a marker that is true only
    sometimes, and "sometimes evaluator-only" is not a side of the boundary. Every such assignment
    is therefore an error and never a quiet "not marked": silence here is the failure mode that
    matters, because the module that meant to be behind the firewall is the module whose contents
    are worth leaking. Only an unconditional module-level binding of the literal ``True`` marks.
    """
    marked: set[str] = set()
    violations: list[Violation] = []
    for name in sorted(modules):
        module = modules[name]
        top_level = {id(stmt) for stmt in module.tree.body}
        for node in ast.walk(module.tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = [node.target]
            else:
                continue
            spelled = sorted({n for t in targets for n in _marker_names(t)})
            if not spelled:
                continue

            simple = (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and id(node) in top_level
                and len(targets) == 1
                and isinstance(targets[0], ast.Name)
            )
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if simple and isinstance(value, ast.Constant) and value.value is True:
                marked.add(name)
                continue
            if not simple:
                violations.append(Violation(
                    module.path, node.lineno, "bad-marker",
                    f"{spelled[0]} must be a plain, unconditional, module-level assignment. Bound "
                    "inside a function, a class body, a branch or a tuple unpack, it is a claim "
                    "the check cannot evaluate, so the module would be silently treated as "
                    "treatment-visible -- the fail-open direction"))
                continue
            violations.append(Violation(
                module.path, node.lineno, "bad-marker",
                f"{spelled[0]} must be assigned the literal True. Any other value leaves it "
                "ambiguous whether this module is evaluator-only, and an ambiguous firewall "
                "boundary is decided by whoever reads it last"))
    return marked, violations


# --------------------------------------------------------------------------------------------
# import graph
# --------------------------------------------------------------------------------------------


def _dotted_prefixes(name: str) -> Iterator[str]:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        yield ".".join(parts[:i])


def _walk_scoped(node: ast.AST, in_function: bool = False) -> Iterator[tuple[ast.AST, bool]]:
    """Yield every descendant with a flag saying whether it sits inside a function body."""
    inside = in_function or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    for child in ast.iter_child_nodes(node):
        yield child, inside
        yield from _walk_scoped(child, inside)


def _relative_base(module: _Module, level: int) -> str | None:
    parts = module.name.split(".")
    if not module.is_package:
        parts = parts[:-1]
    if level > 1:
        if len(parts) < level - 1:
            return None
        parts = parts[: len(parts) - (level - 1)]
    return ".".join(parts) if parts else None


def _edges_of(
    module: _Module,
    known: frozenset[str],
    is_evaluator: Callable[[str], bool],
) -> tuple[list[_Edge], list[Violation]]:
    """Every import edge in one module, including lazy ones and string-named ones.

    A target is kept when it names a module that exists, or when it names evaluator material that
    does not exist yet -- the second case is why a banned name is banned as a *name*.
    """
    edges: list[_Edge] = []
    violations: list[Violation] = []

    def keep(target: str, lineno: int, kind: str) -> None:
        # `import a.b.c` executes a, then a.b, then a.b.c, so every prefix is a real edge. The walk
        # stops at the first evaluator-only prefix: anything below it is a name *inside* that
        # module, and reporting `qrels` and `qrels.TABLE` as two breaches of the same import would
        # double every message for no extra information.
        for candidate in _dotted_prefixes(target):
            if is_evaluator(candidate):
                edges.append(_Edge(module.name, candidate, lineno, kind))
                return
            if candidate in known:
                edges.append(_Edge(module.name, candidate, lineno, kind))

    for node, in_function in _walk_scoped(module.tree):
        kind = "function-local import" if in_function else "import"
        if isinstance(node, ast.Import):
            for alias in node.names:
                keep(alias.name, node.lineno, kind)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from ..evidence.identity import X` inside shapeflow.p1.selectors is an edge to
                # shapeflow.evidence.identity, not to shapeflow: the level resolves the anchor and
                # node.module is appended to it. Dropping the second half would leave most of this
                # repository's imports invisible to the graph, and an invisible import is exactly
                # the one a leak would travel along.
                anchor = _relative_base(module, node.level)
                if anchor is None:
                    violations.append(Violation(
                        module.path, node.lineno, "unresolvable-import",
                        f"relative import of level {node.level} escapes the source tree, so the "
                        "import graph cannot be closed here -- and a firewall proved over an "
                        "incomplete graph proves nothing"))
                    continue
                base = f"{anchor}.{node.module}" if node.module else anchor
            else:
                base = node.module
                if base is None:  # pragma: no cover - the grammar forbids it
                    continue
            keep(base, node.lineno, kind)
            for alias in node.names:
                if alias.name != "*":
                    keep(f"{base}.{alias.name}", node.lineno, kind)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A module name in a string is an import waiting for importlib. It is only an edge when
            # it names something real, so ordinary prose cannot manufacture one.
            text = node.value
            if text and not any(ch.isspace() for ch in text):
                if text in known or is_evaluator(text):
                    edges.append(_Edge(
                        module.name, text, node.lineno, "dynamic module-name reference"))

    # One statement can name the same package several times (`from a.b import c, d` walks the
    # prefix a, a.b twice). Deduplicating keeps a single violation per line instead of one per
    # alias, and the sort keeps the order independent of how the AST was traversed.
    unique = sorted(set(edges), key=lambda e: (e.lineno, e.target, e.kind))
    return unique, violations


def _first_evaluator_chain(
    start: str,
    graph: dict[str, list[_Edge]],
    is_evaluator: Callable[[str], bool],
) -> list[str] | None:
    """Shortest chain from ``start`` to evaluator material, or None.

    Neighbours are visited in sorted order so that two equally short chains always resolve to the
    same one: an error message that changes between runs is an error message people stop trusting.
    """
    if is_evaluator(start):
        return [start]
    seen = {start}
    queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
    while queue:
        node, chain = queue.popleft()
        for target in sorted({e.target for e in graph.get(node, ())}):
            if target in seen:
                continue
            seen.add(target)
            extended = chain + [target]
            if is_evaluator(target):
                return extended
            queue.append((target, extended))
    return None


# --------------------------------------------------------------------------------------------
# answer-key-shaped names
# --------------------------------------------------------------------------------------------


def _identifiers(tree: ast.Module) -> Iterator[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.keyword):
            if node.arg is not None:
                yield node.arg, getattr(node, "lineno", 0) or 0
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                yield name, node.lineno
        elif isinstance(node, ast.alias):
            # `from x import gold_docs as g` binds the answer key under a new name; both halves
            # are worth flagging, and the alias node is where the rename happens.
            yield node.name.rsplit(".", 1)[-1], getattr(node, "lineno", 0) or 0
            if node.asname:
                yield node.asname, getattr(node, "lineno", 0) or 0


def _name_violations(module: _Module, spec: FirewallSpec) -> list[Violation]:
    banned = tuple(spec.forbidden_names) + tuple(spec.forbidden_files)
    found: list[Violation] = []

    def hit(text: str) -> str | None:
        lowered = text.lower()
        for token in banned:
            if token in lowered:
                return token
        return None

    # The module's own dotted name. Every other rule here looks *inside* a file, so a module that
    # is itself called after the material -- `broker/qrels_cache.py`, `p1/gold_docs.py` -- passed
    # clean as long as its body used neutral identifiers, and the treatment package would be
    # carrying the answer key under a name that says so. The import statement does not save this
    # either: `from shapeflow.p1.gold_docs import lookup` puts the module name in `node.module`,
    # which is a plain string on the AST node and not an identifier any walk yields.
    module_token = hit(module.name)
    if module_token is not None:
        found.append(Violation(
            module.path, 1, "answer-key-module-name",
            f"module {module.name!r} is named after evaluator-only material ({module_token!r}) and "
            "is on the treatment path. A module that has to be called this to describe itself "
            "belongs on the evaluator side of the firewall, not inside a treatment package"))

    for name, lineno in _identifiers(module.tree):
        token = hit(name)
        if token is not None:
            found.append(Violation(
                module.path, lineno, "answer-key-name",
                f"identifier {name!r} names evaluator-only material ({token!r}). Gold, qrels, "
                "evidence sets and negatives are evaluator-only wherever they came from -- "
                "handing them in as a plain value is the same leak as importing them"))

    for node in ast.walk(module.tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))):
            continue
        # A record decoded straight from the benchmark's JSONL can be keyed by bytes as easily as
        # by str, so `record[b"gold_docs"]` is the same read as `record["gold_docs"]`. Latin-1
        # never fails, which keeps this from turning an exotic literal into a crash inside the
        # firewall check itself.
        text = (node.value.decode("latin-1")
                if isinstance(node.value, bytes) else node.value)
        # Prose is exempt on purpose: docstrings and messages must be able to *explain* the
        # firewall. A field name, a dict key or a path has no whitespace, and that is the shape
        # this rule is for.
        if not text or any(ch.isspace() for ch in text):
            continue
        token = hit(text)
        if token is None:
            continue
        rule = ("evaluator-data-file"
                if any(f in text.lower() for f in spec.forbidden_files) else "answer-key-name")
        found.append(Violation(
            module.path, node.lineno, rule,
            f"literal {text!r} names evaluator-only material ({token!r}). The decrypted "
            "benchmark file and the label fields inside it are evaluator process material; a "
            "treatment module must not be able to open or address them"))

    found.sort(key=lambda v: (v.lineno, v.rule, v.detail))
    return found


# --------------------------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------------------------


def _in_package(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def check(src_root: Path, spec: FirewallSpec = FREEZE_1) -> Report:
    """Run the firewall check over a source tree.

    Takes the root as a parameter so the check can be pointed at a tree with a planted violation.
    A firewall check that can only ever be run against the real repository can only ever be
    observed passing, which is indistinguishable from a check that cannot fail.
    """
    if not src_root.is_dir():
        return Report(
            (Violation(src_root, 0, "missing-source-root",
                       "source root does not exist, so nothing was scanned. An unrun firewall "
                       "check must never be reported as a clean one"),),
            (), (), tuple(sorted(spec.evaluator_modules)), ())

    modules, violations = _collect(src_root)
    marked, marker_violations = _marked_evaluator(modules)
    violations.extend(marker_violations)

    evaluator_names = frozenset(spec.evaluator_modules) | marked

    def is_evaluator(name: str) -> bool:
        return any(_in_package(name, e) for e in evaluator_names)

    known = frozenset(modules)

    treatment: list[str] = sorted(
        name for name in modules
        if any(_in_package(name, p) for p in spec.treatment_packages))

    # A treatment package that has vanished -- renamed, moved, split -- would silently take its
    # modules out of scope and leave the check reporting success over an empty set.
    for package in sorted(spec.treatment_packages):
        if not any(_in_package(name, package) for name in modules):
            violations.append(Violation(
                src_root, 0, "missing-treatment-package",
                f"treatment package {package!r} has no modules under {src_root}. Either it moved "
                "and this spec is stale, or the firewall is now guarding nothing at all"))

    for name in treatment:
        if is_evaluator(name):
            violations.append(Violation(
                modules[name].path, 0, "treatment-and-evaluator",
                f"{name} is both on the treatment path and evaluator-only. One of the two claims "
                "is wrong, and no import rule can be enforced until a person decides which"))

    graph: dict[str, list[_Edge]] = {}
    for name in sorted(modules):
        edges, edge_violations = _edges_of(modules[name], known, is_evaluator)
        graph[name] = edges
        violations.extend(edge_violations)

    # Direct and transitive reachability, reported at the treatment line that owns the edge --
    # the one line whose author can actually remove it.
    for name in treatment:
        for edge in graph.get(name, ()):
            chain = _first_evaluator_chain(edge.target, graph, is_evaluator)
            if chain is None:
                continue
            evaluator = chain[-1]
            if len(chain) == 1:
                detail = f"{edge.kind} of evaluator-only module {evaluator!r}"
            else:
                route = " -> ".join(chain)
                detail = (f"{edge.kind} of {edge.target!r} reaches evaluator-only {evaluator!r} "
                          f"via {route}")
            violations.append(Violation(
                modules[name].path, edge.lineno, "evaluator-import",
                detail + ". Gold answers, qrels, evidence sets and negatives are evaluator-only: "
                "a treatment path that can read them is scored on its own answer key"))

    # The closure: a module the treatment path imports runs on the treatment path, wherever it
    # lives. Evaluator modules are not traversed -- reaching one is already a violation, and their
    # own bodies are allowed to name the material.
    closure: dict[str, list[str]] = {}
    queue: deque[tuple[str, list[str]]] = deque()
    for name in treatment:
        closure[name] = [name]
        queue.append((name, [name]))
    while queue:
        node, chain = queue.popleft()
        for target in sorted({e.target for e in graph.get(node, ())}):
            if target in closure or target not in modules or is_evaluator(target):
                continue
            closure[target] = chain + [target]
            queue.append((target, chain + [target]))

    on_treatment = set(treatment)
    for name in sorted(closure):
        for violation in _name_violations(modules[name], spec):
            if name in on_treatment:
                violations.append(violation)
            else:
                route = " -> ".join(closure[name])
                violations.append(Violation(
                    violation.path, violation.lineno, violation.rule,
                    violation.detail + f" (on the treatment path via {route})"))

    violations.sort(key=lambda v: (str(v.path), v.lineno, v.rule, v.detail))
    return Report(
        tuple(violations),
        tuple(treatment),
        tuple(sorted(closure)),
        tuple(sorted(evaluator_names)),
        tuple(sorted(n for n in evaluator_names if n in modules)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "src_root", nargs="?", default=str(REPO / "src"),
        help="source tree to scan (default: the repository's src/)")
    args = parser.parse_args(argv)

    report = check(Path(args.src_root))
    if not report.ok:
        print("LEAKAGE FIREWALL CHECK FAILED", file=sys.stderr)
        for violation in report.violations:
            print(f"  {violation.render(REPO)}", file=sys.stderr)
        print(
            "\nSection 3.1 of the protocol: qrels, gold and negative annotations enter the "
            "evaluator\nprocess only. Regression gate 9 is the verification that the firewall "
            "holds. Fix the\nreference -- do not add an exemption, and do not move the data "
            "behind a wrapper that\nlaunders the name.",
            file=sys.stderr,
        )
        return 1

    print(f"leakage firewall: {len(report.treatment_modules)} treatment module(s), "
          f"{len(report.closure_modules)} in closure, "
          f"{len(report.evaluator_modules)} evaluator-only module name(s) "
          f"({len(report.evaluator_on_disk)} present on disk), "
          "no path from treatment to evaluator material")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
