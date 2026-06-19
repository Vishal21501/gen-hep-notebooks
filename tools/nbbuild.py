"""Tiny helpers for building the teaching notebooks with nbformat.

Authoring contract (so the cadence converter works in *auto mode* — headings +
`# cadence:starter` as the ONLY cadence syntax):

* `md(...)`  -> a markdown cell. Use `## ...` headings before every exercise.
* `code(...)` -> a code cell.
* `exercise(...)` -> a code cell that ENDS on a bare answer-variable line, with a
  `# cadence:starter`/`# cadence:end` scaffold inside. Auto mode reads that
  trailing variable's value from the kernel and registers a checkpoint.
* `setup(...)` -> a code cell that ends on a non-answer statement (a `print`),
  so auto mode classifies it as setup and copies it verbatim to students.

Keeping these distinct is the whole discipline: exercises end on a primitive,
setup ends on a print.
"""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def starter(body):
    """Wrap `body` in cadence starter markers. `body` is the scaffold the
    student receives (comments + `... ` placeholders)."""
    return "# cadence:starter\n" + body.strip("\n") + "\n# cadence:end"


def exercise(scaffold_body, solution_body, answer_var):
    """An exercise cell.

    Layout: starter scaffold (what the student fills) -> the teacher's reference
    solution -> a final bare `answer_var` line (the value auto mode checks).

    Anything the student needs *before* this cell (loaded data, trained models,
    helper functions) must live in a `setup()` cell placed BEFORE the exercise's
    heading — using only the starter marker, code outside the starter region is
    stripped from the student notebook, so it can't double as shared setup.
    """
    parts = [
        starter(scaffold_body),
        solution_body.strip("\n"),
        answer_var,  # bare trailing answer -> the checkpoint value auto mode reads
    ]
    return new_code_cell("\n\n".join(parts))


def setup(text):
    """A setup cell — make sure it ends on a `print(...)` so auto mode keeps it
    as verbatim setup rather than treating its last value as an answer."""
    return new_code_cell(text.strip("\n"))


def build(path, cells, title=None, wire_cadence=True):
    """Write the teacher notebook. With `wire_cadence` (default), prepend
    `%load_ext cadence` (just after the title markdown) and append a
    `%cadence_autoregister` cell — the only two cadence *magics*; everything in
    between stays a plain teaching notebook with headings + starter markers."""
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    cells = list(cells)
    if wire_cadence:
        load = new_code_cell(
            "%load_ext cadence\n"
            "# Loads the cadence magics + the input transformer that comments out\n"
            "# `# cadence:starter` regions, so the scaffold placeholders never run."
        )
        # Sit just below a leading title markdown if there is one.
        cells.insert(1 if cells and cells[0].cell_type == "markdown" else 0, load)
        cells.append(new_markdown_cell(
            "## Generate the cadence notebooks\n\n"
            "Run this **after** *Run All* above, so every exercise's answer value is "
            "in the kernel. `%cadence_autoregister` auto-detects each `##`-headed "
            "exercise, registers a checkpoint from the cell's final value, and writes "
            "`<this>_registered.ipynb` — which in turn writes the student notebook via "
            "the `%cadence_scaffold` cell it appends."
        ))
        cells.append(new_code_cell("%cadence_autoregister"))
    nb.cells = cells
    nbf.write(nb, str(path))
    print(f"wrote {path}  ({len(cells)} cells)")
    return nb
