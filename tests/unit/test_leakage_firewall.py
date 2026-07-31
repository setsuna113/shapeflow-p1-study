"""The leakage firewall check must be able to reject.

Regression gate 9 is the only thing standing between "the treatment path cannot see the answer key"
and "nobody has checked lately". A check that has only ever been observed passing is indistinguish-
able from a check that cannot fail, so every rule below is exercised on a *planted* violation: a
direct import of the qrels module, a transitive one through an innocent-looking helper, a lazy
import inside a function body, a dynamic one by string, and the answer-key-shaped names that carry
the same material without importing anything.

The trees are synthetic and tiny. They mirror the real package names because the tests run the real
frozen spec -- testing the check against a made-up configuration would prove the algorithm works
and leave the actual firewall untested.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECK_PATH = REPO / "tools" / "ci" / "check_leakage_firewall.py"


def _load_check():
    """Import the CI script by path.

    ``tools/ci`` is deliberately not a package -- the checks are run as scripts by
    ``scripts/verify_local.sh`` and must not become importable from the library. Registering the
    module in ``sys.modules`` before executing it is required: ``@dataclass`` looks its own module
    up there while it builds the class and raises if the entry is missing.
    """
    spec = importlib.util.spec_from_file_location("shapeflow_ci_leakage_firewall", CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fw = _load_check()


# --------------------------------------------------------------------------------------------
# synthetic trees
# --------------------------------------------------------------------------------------------

#: Every package the real spec names, so the "a treatment package went missing" guard stays quiet
#: unless a test is deliberately aiming at it.
_BASE_PACKAGES = (
    "shapeflow",
    "shapeflow/bench",
    "shapeflow/bench/bcplus",
    "shapeflow/bench/grading",
    "shapeflow/broker",
    "shapeflow/evidence",
    "shapeflow/odr",
    "shapeflow/p1",
    "shapeflow/strategies",
)

#: The real answer-key module, reduced to the two properties the firewall keys on: the marker and
#: the oracle-shaped field names. It is evaluator-only, so its own use of those names is legal --
#: a test that did not include it could not tell "outside the closure" from "not scanned".
_QRELS = '''
"""Evaluator-only: the answer key."""

EVALUATOR_ONLY = True


def load_qrels(path):
    return {"gold_docs": [], "negative_docs": []}
'''


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def _plant(
    tmp_path: Path,
    files: dict[str, str] | None = None,
    *,
    omit_packages: tuple[str, ...] = (),
    with_qrels: bool = True,
) -> Path:
    """Build a source tree that mirrors the real package layout, plus whatever is planted in it."""
    src = tmp_path / "src"
    for package in _BASE_PACKAGES:
        if package in omit_packages:
            continue
        _write(src, f"{package}/__init__.py", '"""package."""\n')
    if with_qrels:
        _write(src, "shapeflow/bench/bcplus/qrels.py", _QRELS)
    for rel, text in sorted((files or {}).items()):
        _write(src, rel, text)
    return src


def _lineno(path: Path, needle: str) -> int:
    """Line of the first occurrence, so no test hardcodes a number a reflow would invalidate."""
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in {path}")


def _rules(report) -> list[str]:
    return sorted({v.rule for v in report.violations})


def _details(report, rule: str) -> str:
    return "\n".join(v.detail for v in report.violations if v.rule == rule)


# --------------------------------------------------------------------------------------------
# the check clears a clean tree (otherwise every rejection below proves nothing)
# --------------------------------------------------------------------------------------------


def test_clean_tree_passes_and_reports_what_it_scanned(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/helpers.py": "VALUE = 1\n",
        "shapeflow/p1/selectors.py": """
            from shapeflow.helpers import VALUE
            from shapeflow.bench.bcplus import splits


            def select(items):
                return sorted(items)[:VALUE]
            """,
        "shapeflow/bench/bcplus/splits.py": "SPLIT = 'itd'\n",
    })
    report = fw.check(src)

    assert report.ok, [v.render() for v in report.violations]
    assert "shapeflow.p1.selectors" in report.treatment_modules
    # The helper the treatment path imports is scanned as part of the treatment path.
    assert "shapeflow.helpers" in report.closure_modules
    assert "shapeflow.bench.bcplus.qrels" in report.evaluator_modules


def test_the_evaluator_module_may_use_its_own_answer_key_names(tmp_path):
    """qrels itself is full of gold_docs. Flagging it would make the check unusable, and the rule
    is about who may *reach* the material, not about who may name it inside the vault."""
    report = fw.check(_plant(tmp_path))
    assert report.ok, [v.render() for v in report.violations]
    assert "shapeflow.bench.bcplus.qrels" not in report.closure_modules


# --------------------------------------------------------------------------------------------
# imports: direct, transitive, lazy, typing-only, dynamic
# --------------------------------------------------------------------------------------------


def test_direct_import_of_the_answer_key_is_rejected(tmp_path):
    # The imported symbol is deliberately neutrally named, so this test proves the *import* rule
    # fires rather than the name rule catching `load_qrels` by accident.
    src = _plant(tmp_path, {
        "shapeflow/p1/aggregators.py": """
            from shapeflow.bench.bcplus.qrels import TABLE


            def aggregate(spans, query_id):
                return TABLE[query_id]
            """,
    })
    report = fw.check(src)

    assert not report.ok
    assert _rules(report) == ["evaluator-import"]
    offender = src / "shapeflow/p1/aggregators.py"
    assert report.violations[0].path == offender
    assert report.violations[0].lineno == _lineno(offender, "import TABLE")
    assert "shapeflow.bench.bcplus.qrels" in report.violations[0].detail


def test_an_answer_key_symbol_trips_both_rules(tmp_path):
    """`from ... import load_qrels` is two separate breaches on one line: the module is reachable,
    and the name is bound into this module's namespace. Both are reported, because removing the
    import and keeping a locally-defined `load_qrels` would fix neither."""
    src = _plant(tmp_path, {
        "shapeflow/p1/aggregators.py": "from shapeflow.bench.bcplus.qrels import load_qrels\n",
    })
    assert _rules(fw.check(src)) == ["answer-key-name", "evaluator-import"]


def test_import_of_the_containing_package_is_rejected_by_name_when_the_file_is_absent(tmp_path):
    """The ban is on the name, not on the file.

    `qrels` lands and leaves and is renamed while the study is being built. If the check only
    rejected imports that currently resolve, the window in which the module does not exist would be
    a window in which writing the import is legal -- and the import would still be there when the
    module came back.
    """
    src = _plant(tmp_path, {
        "shapeflow/strategies/page_h.py": """
            from shapeflow.bench.bcplus.qrels import TABLE


            def rank(pages):
                return TABLE
            """,
    }, with_qrels=False)
    report = fw.check(src)

    assert _rules(report) == ["evaluator-import"]
    assert "shapeflow.bench.bcplus.qrels" in _details(report, "evaluator-import")
    assert "shapeflow.bench.bcplus.qrels" not in report.evaluator_on_disk


def test_transitive_import_is_rejected_and_names_the_route(tmp_path):
    """The middle module is the whole problem: it looks like a data helper and it is not."""
    src = _plant(tmp_path, {
        "shapeflow/bench/bcplus/relevance.py": """
            from shapeflow.bench.bcplus.qrels import LOOKUP

            TABLE = LOOKUP
            """,
        "shapeflow/broker/features.py": """
            from shapeflow.bench.bcplus.relevance import TABLE


            def feature(job):
                return TABLE
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["evaluator-import"]
    offender = src / "shapeflow/broker/features.py"
    (violation,) = report.violations
    assert violation.path == offender
    assert violation.lineno == _lineno(offender, "import TABLE")
    assert ("shapeflow.bench.bcplus.relevance -> shapeflow.bench.bcplus.qrels"
            in violation.detail)


def test_function_local_import_is_rejected_and_named_as_such(tmp_path):
    """A lazy import is how this rule actually gets broken: it never shows up in the import block,
    it looks temporary, and it runs on exactly the path that matters."""
    src = _plant(tmp_path, {
        "shapeflow/p1/preflight.py": """
            def preflight(candidate):
                from shapeflow.bench.bcplus.qrels import TABLE

                return TABLE
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["evaluator-import"]
    (violation,) = report.violations
    assert violation.lineno == _lineno(src / "shapeflow/p1/preflight.py", "import TABLE")
    assert "function-local import" in violation.detail


def test_typing_only_import_is_rejected(tmp_path):
    """`if TYPE_CHECKING:` does not execute, but it is one deleted line away from an import that
    does, and it means the type of the answer key is already part of this module's interface."""
    src = _plant(tmp_path, {
        "shapeflow/evidence/lineage.py": """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from shapeflow.bench.bcplus.qrels import Judgements


            def lineage(x: "Judgements") -> str:
                return str(x)
            """,
    })
    assert _rules(fw.check(src)) == ["evaluator-import"]


def test_dynamic_import_by_string_is_rejected(tmp_path):
    """importlib leaves no Import node. A module name in a literal is an import waiting to happen,
    and treating it as an edge is the only way the graph stays closed."""
    src = _plant(tmp_path, {
        "shapeflow/odr/adapter.py": """
            import importlib


            def load():
                return importlib.import_module("shapeflow.bench.bcplus.qrels")
            """,
    })
    report = fw.check(src)

    # Two rules on one line, and both are true: the literal is an import edge, and a literal that
    # spells out the answer key's module is answer-key-shaped whatever it is passed to.
    assert _rules(report) == ["answer-key-name", "evaluator-import"]
    assert "dynamic module-name reference" in _details(report, "evaluator-import")


def test_dynamic_import_inside_a_helper_is_rejected_transitively(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/loaders.py": """
            import importlib


            def load(name="shapeflow.bench.bcplus.qrels"):
                return importlib.import_module(name)
            """,
        "shapeflow/strategies/pipeline.py": """
            from shapeflow.loaders import load


            def run():
                return load()
            """,
    })
    report = fw.check(src)

    # The helper is inside the closure, so its literal is flagged there too; the violation that
    # matters is the one on the treatment module, which is where the route is printed.
    imports = [v for v in report.violations if v.rule == "evaluator-import"]
    (violation,) = imports
    assert violation.path == src / "shapeflow/strategies/pipeline.py"
    assert "shapeflow.loaders -> shapeflow.bench.bcplus.qrels" in violation.detail


def test_marker_alone_makes_a_module_evaluator_only(tmp_path):
    """A new evaluator module must be able to join the firewall by saying so in its own file. If
    the only way in were this check's hardcoded list, the module that nobody remembered to add is
    the one that leaks."""
    src = _plant(tmp_path, {
        "shapeflow/bench/bcplus/hard_negatives.py": """
            \"\"\"Evaluator-only.\"\"\"

            EVALUATOR_ONLY = True

            NEGATIVES = {}
            """,
        "shapeflow/strategies/prose.py": """
            from shapeflow.bench.bcplus.hard_negatives import NEGATIVES


            def prose(pages):
                return NEGATIVES
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["evaluator-import"]
    assert "shapeflow.bench.bcplus.hard_negatives" in report.evaluator_modules


def test_marker_must_be_the_literal_true(tmp_path):
    """`EVALUATOR_ONLY = os.environ.get(...)` would make the boundary a runtime question. The check
    refuses to guess which way it resolves."""
    src = _plant(tmp_path, {
        "shapeflow/bench/bcplus/hard_negatives.py": """
            import os

            EVALUATOR_ONLY = bool(os.environ.get("EVAL"))
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["bad-marker"]
    assert "literal True" in _details(report, "bad-marker")


def test_marker_set_to_false_does_not_quietly_unmark_a_module(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/bench/bcplus/hard_negatives.py": "EVALUATOR_ONLY = False\n",
    })
    assert _rules(fw.check(src)) == ["bad-marker"]


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("conditional", "if True:\n    EVALUATOR_ONLY = True\n\nSECRET = {}\n"),
        ("try-block", "try:\n    EVALUATOR_ONLY = True\nexcept Exception:\n    pass\n\n"
                      "SECRET = {}\n"),
        ("class-body", "class Meta:\n    EVALUATOR_ONLY = True\n\n\nSECRET = {}\n"),
        ("tuple-unpack", "EVALUATOR_ONLY, VERSION = True, 1\n\nSECRET = {}\n"),
        ("function-local", "def mark():\n    global EVALUATOR_ONLY\n    EVALUATOR_ONLY = True\n\n\n"
                           "SECRET = {}\n"),
    ],
)
def test_a_marker_that_is_not_a_plain_module_level_assignment_is_rejected(tmp_path, label, body):
    """The fail-open direction is the one that matters.

    A module that *meant* to declare itself evaluator-only is a module whose contents are worth
    leaking. If the check silently reads a conditional or unpacked marker as "no marker", the
    author sees the declaration in their own file, the firewall sees ordinary treatment-visible
    code, and a treatment module may import it with no violation at all -- which is exactly what
    happened for every case below before this rule existed.
    """
    src = _plant(tmp_path, {
        "shapeflow/bench/bcplus/vault.py": body,
        "shapeflow/p1/consumer.py": """
            from shapeflow.bench.bcplus.vault import SECRET

            USE = SECRET
            """,
    })
    report = fw.check(src)

    assert "bad-marker" in _rules(report), f"{label} marker was accepted in silence"
    assert "fail-open" in _details(report, "bad-marker")


# --------------------------------------------------------------------------------------------
# answer-key-shaped names: the leak that imports nothing
# --------------------------------------------------------------------------------------------


def test_answer_key_attribute_access_is_rejected(tmp_path):
    """The material can arrive as a plain dict passed in from above. Reading `.gold_docs` off it is
    the same leak as importing the qrels module, with none of the import graph to catch it."""
    src = _plant(tmp_path, {
        "shapeflow/p1/renderer.py": """
            def render(task):
                return task.gold_docs
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["answer-key-name"]
    assert report.violations[0].lineno == _lineno(src / "shapeflow/p1/renderer.py", "gold_docs")


def test_answer_key_dict_key_is_rejected(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/broker/driver.py": """
            def decide(record):
                return len(record["negative_docs"])
            """,
    })
    assert _rules(fw.check(src)) == ["answer-key-name"]


def test_qrel_shaped_variable_name_is_rejected(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/evidence/manifest.py": """
            def build(root):
                qrels_path = root / "labels.txt"
                return qrels_path
            """,
    })
    assert _rules(fw.check(src)) == ["answer-key-name"]


def test_keyword_argument_named_after_the_answer_key_is_rejected(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/odr/continuation.py": """
            def score(**kwargs):
                return kwargs


            def go():
                return score(answer_key=1)
            """,
    })
    assert _rules(fw.check(src)) == ["answer-key-name"]


@pytest.mark.parametrize("expression", [
    "task.evidence_docs",
    'record["evidence_docs"]',
    "view.evidence_docids",
])
def test_the_evidence_document_set_is_evaluator_only_too(tmp_path, expression):
    """AGENTS.md section 2 names four things: gold answers, qrels, *evidence* and the negatives.

    `evidence_docs` is a field of the decrypted benchmark file and of the qrels view, and reading
    it tells a selector which pages the benchmark considers dispositive. Guarding gold and the
    negatives while leaving the evidence set unnamed made this rule's own failure message -- "gold,
    qrels, evidence sets and negatives are evaluator-only" -- untrue of the check that printed it.
    """
    src = _plant(tmp_path, {
        "shapeflow/p1/aggregators.py": f"""
            def aggregate(task, record, view):
                return {expression}
            """,
    })
    assert _rules(fw.check(src)) == ["answer-key-name"]


def test_the_treatment_path_may_still_talk_about_evidence(tmp_path):
    """`shapeflow.evidence` is a treatment package. Banning the bare word would flag the
    compression path's own name, and a check that flags the thing it is protecting gets removed."""
    src = _plant(tmp_path, {
        "shapeflow/evidence/shared_view.py": """
            def build(evidence_spans, evidence_budget):
                return evidence_spans[:evidence_budget]
            """,
    })
    assert fw.check(src).ok, [v.render(src) for v in fw.check(src).violations]


def test_a_bytes_key_is_the_same_read_as_a_str_key(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/broker/loader.py": """
            def read(record):
                return record[b"gold_docs"]
            """,
    })
    assert _rules(fw.check(src)) == ["answer-key-name"]


def test_a_treatment_module_named_after_the_answer_key_is_rejected(tmp_path):
    """Every other name rule looks inside a file. A module whose own name is the answer key, with
    neutral identifiers in its body, was the one shape that walked through: the import that uses it
    carries the module name in `node.module`, which is a plain string on the AST node and not an
    identifier any walk yields."""
    src = _plant(tmp_path, {
        "shapeflow/broker/qrels_cache.py": """
            TABLE = {}


            def get(key):
                return TABLE.get(key)
            """,
        "shapeflow/broker/driver.py": """
            from shapeflow.broker.qrels_cache import get

            USE = get
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["answer-key-module-name"]
    assert report.violations[0].path == src / "shapeflow/broker/qrels_cache.py"
    assert "shapeflow.broker.qrels_cache" in report.violations[0].detail


def test_a_module_named_after_the_answer_key_outside_the_closure_is_not_flagged(tmp_path):
    """Same scope discipline as every other name rule: the evaluator's own modules are named after
    the material by necessity, and flagging them would train people to ignore the check."""
    src = _plant(tmp_path, {"shapeflow/reporting/gold_docs.py": "TABLE = {}\n"})
    assert fw.check(src).ok, [v.render(src) for v in fw.check(src).violations]


def test_evaluator_data_file_reference_is_rejected(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/strategies/p0.py": """
            from pathlib import Path


            def read(root):
                return Path(root, "browsecomp_plus_decrypted.jsonl").read_text()
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["evaluator-data-file"]
    assert "browsecomp_plus_decrypted" in _details(report, "evaluator-data-file")


def test_prose_may_name_the_material(tmp_path):
    """A firewall nobody can write documentation next to gets ignored, then disabled. Any string
    with whitespace is prose; a field name, a dict key and a path have none."""
    src = _plant(tmp_path, {
        "shapeflow/p1/view.py": '''
            """The view never sees gold_docs, qrels or browsecomp_plus_decrypted.jsonl.

            Those are evaluator-only; the answer_key never reaches this module.
            """


            def view(pages):
                return pages
            ''',
    })
    assert fw.check(src).ok


def test_answer_key_name_in_an_imported_helper_is_rejected_with_its_route(tmp_path):
    """A helper is on the treatment path if the treatment path imports it, whatever directory it
    lives in. The route is printed because otherwise the file's author cannot tell why their module
    is in scope at all."""
    src = _plant(tmp_path, {
        "shapeflow/features.py": """
            def features(record):
                return {"n": len(record.gold_docs)}
            """,
        "shapeflow/broker/reference.py": """
            from shapeflow.features import features


            def decide(record):
                return features(record)
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["answer-key-name"]
    assert "on the treatment path via shapeflow.broker.reference -> shapeflow.features" in (
        _details(report, "answer-key-name"))


def test_answer_key_name_outside_the_closure_is_not_flagged(tmp_path):
    """Scope discipline, for the same reason the observability check is narrow: a check that fires
    on the evaluator's own code trains people to ignore it."""
    src = _plant(tmp_path, {
        "shapeflow/reporting.py": """
            def report(record):
                return record.gold_docs
            """,
    })
    report = fw.check(src)

    assert report.ok, [v.render() for v in report.violations]
    assert "shapeflow.reporting" not in report.closure_modules


# --------------------------------------------------------------------------------------------
# the check fails closed when it cannot certify the tree
# --------------------------------------------------------------------------------------------


def test_unparseable_module_fails_closed(tmp_path):
    src = _plant(tmp_path, {"shapeflow/p1/broken.py": "def f(:\n    pass\n"})
    report = fw.check(src)

    assert "unparseable" in _rules(report)
    assert "cannot certify" in _details(report, "unparseable")


def test_missing_treatment_package_fails_closed(tmp_path):
    """If `shapeflow.p1` is renamed and this spec is not, the check keeps printing success over a
    set of zero modules -- the most dangerous state a firewall can be in."""
    src = _plant(tmp_path, omit_packages=("shapeflow/p1",))
    report = fw.check(src)

    assert _rules(report) == ["missing-treatment-package"]
    assert "shapeflow.p1" in _details(report, "missing-treatment-package")


def test_missing_source_root_fails_closed(tmp_path):
    report = fw.check(tmp_path / "no-such-tree")

    assert _rules(report) == ["missing-source-root"]
    assert not report.treatment_modules


def test_unresolvable_relative_import_fails_closed(tmp_path):
    """An import the check cannot resolve is a hole in the graph, and a firewall proved over a
    graph with a hole in it is not proved."""
    src = _plant(tmp_path, {
        "shapeflow/p1/handle_codec.py": """
            from .... import something

            CODEC = something
            """,
    })
    report = fw.check(src)

    assert _rules(report) == ["unresolvable-import"]
    assert report.violations[0].lineno == _lineno(
        src / "shapeflow/p1/handle_codec.py", "from ....")


def test_module_cannot_be_both_treatment_and_evaluator(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/p1/contracts.py": """
            \"\"\"Claims both sides.\"\"\"

            EVALUATOR_ONLY = True
            """,
    })
    report = fw.check(src)

    assert "treatment-and-evaluator" in _rules(report)
    assert "One of the two claims" in _details(report, "treatment-and-evaluator")


# --------------------------------------------------------------------------------------------
# determinism and the exit-code contract
# --------------------------------------------------------------------------------------------


def test_output_is_byte_identical_across_runs(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/p1/a.py": "from shapeflow.bench.bcplus.qrels import TABLE\n",
        "shapeflow/broker/b.py": """
            def b(record):
                return record.gold_docs
            """,
        "shapeflow/strategies/c.py": """
            def c():
                from shapeflow.bench.bcplus.qrels import TABLE

                return TABLE
            """,
    })
    first = [v.render(src) for v in fw.check(src).violations]
    second = [v.render(src) for v in fw.check(src).violations]

    assert first == second
    # Sorted by path, then line, then rule: dict and glob order must not be able to reorder a
    # failure report, or a re-run "changes" and people start diffing noise.
    assert first == sorted(first)
    assert [v.rule for v in fw.check(src).violations] == [
        "answer-key-name", "evaluator-import", "evaluator-import"]


def test_script_exits_zero_on_a_clean_tree(tmp_path):
    src = _plant(tmp_path)
    result = subprocess.run(
        [sys.executable, str(CHECK_PATH), str(src)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "leakage firewall:" in result.stdout


def test_script_exits_one_and_prints_file_line_and_why(tmp_path):
    src = _plant(tmp_path, {
        "shapeflow/p1/selectors.py": """
            from shapeflow.bench.bcplus.qrels import load_qrels

            SELECT = load_qrels
            """,
    })
    result = subprocess.run(
        [sys.executable, str(CHECK_PATH), str(src)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "LEAKAGE FIREWALL CHECK FAILED" in result.stderr
    assert f"selectors.py:{_lineno(src / 'shapeflow/p1/selectors.py', 'import load_qrels')}:" in (
        result.stderr)
    assert "scored on its own answer key" in result.stderr
    # The remedy must not read as "add an exemption".
    assert "do not add an exemption" in result.stderr


# --------------------------------------------------------------------------------------------
# the real tree, and the wiring that makes any of this run
# --------------------------------------------------------------------------------------------


def test_repository_tree_is_clean_today(tmp_path):
    report = fw.check(REPO / "src")

    assert report.ok, "\n".join(v.render(REPO) for v in report.violations)
    assert report.treatment_modules, "no treatment modules were scanned at all"
    assert "shapeflow.bench.bcplus.qrels" in report.evaluator_modules


def test_check_is_wired_into_verify_local():
    """An unwired check is a file, not a gate."""
    script = (REPO / "scripts" / "verify_local.sh").read_text(encoding="utf-8")

    assert "tools/ci/check_leakage_firewall.py" in script
    # Ordering matters only in that the existing checks were not moved to make room.
    assert script.index("tools/ci/check_observability.py") < script.index(
        "tools/ci/check_leakage_firewall.py")


@pytest.mark.parametrize("package", fw.FREEZE_1.treatment_packages)
def test_every_declared_treatment_package_exists(package):
    """The spec names packages by string. A rename that lands without touching this file would
    silently empty the scan set, and the check would go on printing success."""
    path = REPO / "src" / Path(*package.split("."))

    assert path.is_dir(), f"{package} is in the firewall spec but not in the tree"
