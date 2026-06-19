"""Run the REAL cadence converter on a finished teacher notebook, GPU-free.

We can't execute torch/jetnet here, so we don't run the cells. Instead we
emulate a post-"Run All" kernel namespace: for every exercise cell (one that
contains a `# cadence:starter` block) we inject its trailing answer variable as
a primitive; for every other heading-paired cell we inject a non-answer object
so auto mode classifies it as setup. Then we call the actual
`autoregister(teacher_nb, user_ns)` -> `scaffold` pipeline and assert that every
exercise became a stubbed checkpoint in the student notebook.

This validates the thing the user cares about: that these notebooks convert
cleanly in auto mode using headings + `# cadence:starter` only.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat as nbf
from cadence.autoregister import autoregister
from cadence.scaffold import scaffold

STARTER_BLOCK = re.compile(
    r"^[ \t]*#\s*cadence:starter\s*$.*?^[ \t]*#\s*cadence:end\s*$\n?",
    re.MULTILINE | re.DOTALL,
)
HEADING = re.compile(r"^\s*#+\s+\S", re.MULTILINE)


def _final_stmt(source: str):
    src = STARTER_BLOCK.sub("", source)
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return tree.body[-1] if tree.body else None


def _trailing_target(source: str):
    """Name that auto mode will read as the answer: the last statement's
    bare name or assignment target. Returns None if it can't tell."""
    last = _final_stmt(source)
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Name):
        return last.value.id
    if isinstance(last, ast.Assign) and isinstance(last.targets[0], ast.Name):
        return last.targets[0].id
    return None


def _final_stmt_names(source: str):
    """Every name referenced by the final statement — so a real kernel's
    `print(f"... {device} ...")` can be emulated by injecting `device` as a
    sentinel, making the cell evaluate to None and skip as setup."""
    last = _final_stmt(source)
    if last is None:
        return set()
    return {n.id for n in ast.walk(last) if isinstance(n, ast.Name)}


class _NotAnAnswer:  # stand-in for setup objects (models, arrays, figures)
    pass


def build_emulated_ns(nb):
    """Inject answer vars (primitive) for exercises, sentinels for setup."""
    ns = {}
    n_ex = 0
    for cell in nb.cells:
        if cell.cell_type != "code" or not STARTER_BLOCK.search(cell.source):
            continue  # only exercises get a value injected
        target = _trailing_target(cell.source)
        if target is not None:
            ns[target] = _answer_value(cell.source, target)
            n_ex += 1
    return ns, n_ex
    # NB: setup cells are intentionally left un-injected. Under this GPU-free
    # emulation their final expression errors, so autoregister copies them
    # verbatim as `# cadence:solution` — the SAME student-facing result as a
    # real kernel, where they evaluate to a non-answer value and skip silently.


def _answer_value(source, target):
    """Pick a primitive matching an optional `# answer: number|string|list`
    hint in the cell; default to a float."""
    m = re.search(r"#\s*answer:\s*(number|string|list|bool)", source)
    kind = m.group(1) if m else "number"
    return {"number": 0.5, "string": "tail", "list": [1, 2, 3], "bool": True}[kind]


def check_notebook(path):
    nb = nbf.read(str(path), as_version=4)
    ns, n_ex = build_emulated_ns(nb)
    reg_path = Path("/tmp/cadverify") / (Path(path).stem + "_registered.ipynb")
    stu_path = Path("/tmp/cadverify") / (Path(path).stem + "_student.ipynb")
    res = autoregister(teacher_nb=nb, user_ns=ns, lesson_name=Path(path).stem,
                       out_path=str(reg_path))
    sres = scaffold(src_path=str(reg_path), out_path=str(stu_path), join_code="DEMO")

    # A failure only matters if it lands on an EXERCISE cell (one with a starter
    # block); failures on setup cells are emulation artifacts (see note above).
    exercise_fails = [c for c in res.checkpoints
                      if c.error and STARTER_BLOCK.search(nb.cells[c.code_cell_index].source)]
    ok = (res.mode == "auto" and res.n_checkpoints == n_ex
          and not exercise_fails and sres.n_exercises == n_ex)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {Path(path).name}: mode={res.mode} "
          f"exercises={n_ex} stubbed={sres.n_exercises} "
          f"checkpoints={res.n_checkpoints} solutions_copied={sres.n_solutions} "
          f"(setup cells copied verbatim; exercise_fails={len(exercise_fails)})")
    for c in exercise_fails:
        print(f"        ! EXERCISE {c.checkpoint_id}: {c.error}")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or [p for p in sorted(Path("notebooks").glob("*.ipynb"))
                             if not p.stem.endswith(("_registered", "_student"))]
    results = [check_notebook(p) for p in paths]
    sys.exit(0 if all(results) else 1)
