# Vendoring notes: JFBench

This directory vendors a subset of [JFBench](https://github.com/pfnet-research/jfbench)
(Japanese instruction Following Benchmark, by Preferred Networks, Inc.), for use as an
importable library inside `lighteval`. No lighteval task/metric definitions live here —
only the upstream constraint library and benchmark-data builder.

- **Upstream repository:** https://github.com/pfnet-research/jfbench
- **Vendored commit:** `bbaf1335b3e4ee94356fc80835137111fe9476eb`
- **Upstream license:** MIT (Copyright (c) 2026 Preferred Networks, Inc.), copied verbatim
  to `LICENSE-MIT-jfbench.txt` in this directory.
- **Bundled data licenses:** the file `data/ifbench_ja_translated.jsonl` and its README
  carry their own licenses, copied verbatim from upstream: `data/LICENSE-CC-BY-SA-4.0`,
  `data/LICENSE-ODC-BY-1.0`, `data/README.md`. See that README for which license applies
  to which part of the data.

## What was vendored

From upstream `src/jfbench/`:

- `__init__.py`, `_data.py`, `imports.py`, `protocol.py`, `llm.py`
- `constraints/` — entire tree verbatim (module code, docstrings, instruction wording,
  competitive-constraint tables, and group-name resolution logic untouched):
  `_competitives.py`, `_group.py`, `_utils.py`, `__init__.py`, `README.md`, and all
  subpackages (`character`, `content`, `format`, `ifbench_count`, `ifbench_format`,
  `ifbench_ratio`, `ifbench_repeat`, `ifbench_sentence`, `ifbench_words`, `length`,
  `logic`, `meta_output`, `notation`, `processing`, `structure`, `style`)
- `prompts/__init__.py`, `prompts/ifbench.py` (with the ja_stackoverflow prompt source
  excluded — see deviations below)
- `benchmark/build.py` (sampling logic untouched); `benchmark/__init__.py` is new,
  created empty (upstream's `benchmark/` had no `__init__.py`)
- `data/ifbench_ja_translated.jsonl` (300 lines), `data/README.md`,
  `data/LICENSE-CC-BY-SA-4.0`, `data/LICENSE-ODC-BY-1.0`

Explicitly **not** vendored (per instructions — out of scope for this package):
`sft_dataset/`, `dpo_dataset/`, `grpo_dataset/`, `visualization/`, `randomize/`,
`benchmark/eval.py`, `benchmark/analyze.py`, `tests/`, `prompts/ja_stackoverflow.py`,
`data/ja_stackoverflow_train.jsonl.zst`.

No lighteval task definitions or metrics are included in this vendored package; those
are maintained separately elsewhere in the `swallow` tasks tree.

## Deviations from upstream

All deviations are also marked in-place in the code with a
`# swallow-evaluation-instruct:` comment.

1. **Import rewriting.** All absolute imports rooted at `jfbench` (e.g.
   `from jfbench.constraints.format import JsonFormatConstraint`,
   `from jfbench._data import DATA_DIR`) were rewritten to the vendored package root
   `lighteval.tasks.swallow.jfbench._vendor_jfbench` via a scripted `sed` pass, verified
   by grepping for any remaining bare `jfbench.` module references afterward (none
   found). `constraints/_group.py`'s `ConstraintGroupMixin.to_serializable_dict` /
   `from_serializable_dict` round-trip constraint classes through
   `cls.__module__` / `importlib.import_module(...)` dynamically (not a hardcoded
   string), so it needed no edit and works correctly with the new module paths — its
   group-name resolution (`_extract_group_name` / `_extract_group_from_module`) still
   finds the `"constraints"` path/module segment because that segment is preserved
   under the new dotted path.

2. **`prompts/__init__.py`** was trimmed to only import/export `IFBenchPrompt` and
   `get_all_ifbench_prompts`, since `ja_stackoverflow.py` (and its
   `get_all_ja_stackoverflow_prompts` / `JaStackoverflowPrompt`) is out of scope and not
   vendored. `benchmark/build.py` and the rest of the vendored tree were grepped for any
   reference to `ja_stackoverflow`/`zstandard`/`JaStackoverflowPrompt`/`zstd`; none were
   found, so no further edits were needed there.

3. **`prompts/ifbench.py`** no longer uses `pandas.read_json(...)` to load the bundled
   JSONL. It was replaced with plain `json` + line-by-line `open()` parsing that builds
   the identical list of `IFBenchPrompt` objects from the `japanese_prompt_without_constraints`
   column (`JA_PROMPT_COL`). The public API (`IFBenchPrompt`, `get_all_ifbench_prompts`,
   `DATA_PATH`, `JA_PROMPT_COL`) and the exact prompt `text()` template are unchanged.
   This drops the `pandas` dependency for the vendored package.

4. **`_data.py`** keeps the `JFBENCH_DATA_DIR` env override and the package-relative
   `data/` fallback, but drops upstream's additional fallback that walked up parent
   directories looking for a `pyproject.toml` to anchor a `data` dir. Inside this repo
   that walk would have escaped `_vendor_jfbench` and resolved to an unrelated directory
   in the host project, so it was removed; the fallback is now unconditionally the
   package-relative `_vendor_jfbench/data` directory.

5. **Python 3.11 compatibility.** Upstream targets Python >=3.12. The vendored tree was
   checked for PEP 695 `type X = ...` statements, `itertools.batched`, and generic
   `class Foo[T]` / `def foo[T](...)` syntax — none were found anywhere in the vendored
   files, so no compatibility rewrite was necessary. `zip(..., strict=True)` usage (fine
   on 3.11) was left untouched. Verified by `python3.11 -m compileall` succeeding
   (exit 0) over the whole vendored tree.

6. **`benchmark/__init__.py`** did not exist upstream (`benchmark/` was an implicit
   namespace-style directory with no package init); an empty `__init__.py` was added so
   `benchmark` is an explicit regular package under `_vendor_jfbench`.

No changes were made to constraint evaluation logic, instruction wording, the
competitive-constraint tables (`_competitives.py`), or `benchmark/build.py`'s sampling
logic — faithfulness to upstream was the priority for anything that affects benchmark
scores.
