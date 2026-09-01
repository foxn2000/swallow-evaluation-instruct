# Vendored BFCL v4 evaluation code

This directory vendors the minimum subset of the Berkeley Function Calling
Leaderboard (BFCL) v4 evaluation code needed to:

1. build the prompting-mode system prompt from a BFCL test entry
   (`formulate_system_prompt` + `_func_doc_language_specific_pre_processing`),
2. decode a model's textual response into function calls
   (`default_decode_ast_prompting` / `ast_parse`), and
3. score a decoded response with the AST checker / relevance checker
   (`ast_checker` and friends).

It intentionally does **not** include any lighteval task definitions, model
handlers, the executable/REST checker, the multi-turn checker, or the model
registry (`MODEL_CONFIG_MAPPING`) — those are out of scope for this vendoring
pass.

## Source

- Upstream repo: https://github.com/ShishirPatil/gorilla
- Upstream package: `berkeley-function-call-leaderboard/bfcl_eval`
- Commit vendored from: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- License: Apache-2.0 (see `LICENSE-Apache-2.0.txt` in this directory,
  copied verbatim from the upstream repo root).

## File-by-file provenance

| Vendored file | Upstream source | Verbatim? |
|---|---|---|
| `enums.py` | `bfcl_eval/constants/enums.py` | Yes |
| `type_mappings.py` | `bfcl_eval/constants/type_mappings.py` | Yes |
| `default_prompts.py` | `bfcl_eval/constants/default_prompts.py` | Yes |
| `java_type_converter.py` | `bfcl_eval/eval_checker/ast_eval/type_convertor/java_type_converter.py` | Yes (import rewritten) |
| `js_type_converter.py` | `bfcl_eval/eval_checker/ast_eval/type_convertor/js_type_converter.py` | Yes (import rewritten) |
| `java_parser.py` | `bfcl_eval/model_handler/parser/java_parser.py` | Function bodies verbatim; `tree_sitter` init made lazy (see below) |
| `js_parser.py` | `bfcl_eval/model_handler/parser/js_parser.py` | Function bodies verbatim; `tree_sitter` init made lazy (see below) |
| `json_parser.py` | `bfcl_eval/model_handler/parser/json_parser.py` | Yes |
| `xml_parser.py` | `bfcl_eval/model_handler/parser/xml_parser.py` | Yes |
| `ast_checker.py` | `bfcl_eval/eval_checker/ast_eval/ast_checker.py` | Function bodies verbatim; `model_name`/`MODEL_CONFIG_MAPPING` replaced with `underscore_to_dot: bool` (see below) |
| `parse_utils.py` | Hand-extracted subset of `bfcl_eval/model_handler/utils.py` and `bfcl_eval/utils.py` | Each included function's body is verbatim; see the module docstring in the file for exactly which functions were pulled in and why, and which were deliberately left out |

`__init__.py` files (this package's, and both upstream `type_convertor/` and
`parser/` packages') are empty in upstream, so this package's `__init__.py`
only carries a short docstring — not a functional deviation.

## Deviations from upstream (all prefixed `# swallow-evaluation-instruct:` in
the code)

1. **Import rewriting.** Every `from bfcl_eval...` / `import bfcl_eval...`
   was rewritten to a relative import within this package (e.g.
   `from .enums import Language`). There are zero remaining references to
   `bfcl_eval` in any importable code path in this directory.

2. **`ast_checker.py`: dropped the model registry dependency.**
   Upstream's `convert_func_name(function_name, model_name: str)` looked up
   `MODEL_CONFIG_MAPPING[model_name.replace("_", "/")].underscore_to_dot` to
   decide whether to replace `"."` with `"_"` in a function name (OpenAI /
   Mistral / Google don't allow dots in function names). We do not vendor the
   model registry. Instead:
   - `convert_func_name(function_name, model_name: str)` became
     `convert_func_name(function_name, underscore_to_dot: bool = False)`,
     keeping the same body semantics (replace `.` with `_` only when
     `underscore_to_dot` is true and the name contains a `.`).
   - The `model_name: str` parameter was replaced with
     `underscore_to_dot: bool = False` in `ast_checker`,
     `simple_function_checker`, `parallel_function_checker_enforce_order`,
     `parallel_function_checker_no_order`, and `multiple_function_checker`,
     threading the boolean through instead of the model name string. No
     other checker logic was changed. Callers must resolve
     `underscore_to_dot` themselves (e.g. from their own model config) before
     calling into this module.

3. **`java_parser.py` / `js_parser.py`: lazy `tree_sitter` initialization.**
   Upstream constructs the `tree_sitter.Language`/`Parser` singleton at
   *module import time*. That would make `import _vendor_bfcl` (or even just
   `import _vendor_bfcl.ast_checker`, since `parse_utils.py` imports the
   parser modules) hard-require `tree_sitter` + `tree_sitter_java` /
   `tree_sitter_javascript` to be installed, even for purely Python BFCL
   categories that never call the Java/JS parser. We moved the
   `Language(...)`/`Parser()` construction into a module-level cached getter
   (`_get_parser()`), called lazily from `parse_java_function_call` /
   `parse_javascript_function_call` on first use. The parsing logic itself
   (the tree traversal / argument extraction) is untouched.

4. **`parse_utils.py` is a hand-assembled file, not a 1:1 copy of one
   upstream file.** It combines the transitive closure of what
   `ast_parse`, `default_decode_ast_prompting`, `formulate_system_prompt`,
   and `_func_doc_language_specific_pre_processing` need from
   `bfcl_eval/model_handler/utils.py` and `bfcl_eval/utils.py`, plus
   `is_function_calling_format_output` and `is_empty_output` (defined in
   `bfcl_eval/utils.py` and re-exported through
   `bfcl_eval/eval_checker/eval_runner_helper.py` via `from bfcl_eval.utils
   import *`; there is no separate definition of these two functions in
   `eval_runner_helper.py` itself). Every individual function body inside
   `parse_utils.py` is a verbatim copy from its named upstream source; the
   file's docstring lists precisely what was included and, for
   traceability, what upstream helpers/predicates were deliberately left
   out because they are not on the call graph of the four entry points
   (e.g. `convert_to_tool`, `extract_test_category_from_id`,
   `default_decode_execute_prompting`, multi-turn/memory/agentic helpers).

5. **No `category_mapping.py` was vendored.** The task brief anticipated
   `is_java` / `is_js` / `extract_test_category_from_id` /
   `extract_prompt_format_from_id` might depend on the large constant
   tables in `bfcl_eval/constants/category_mapping.py`. On inspection,
   `is_java` and `is_js` (the two of these actually needed by the four
   entry points) are plain substring checks with no table lookups, so
   nothing from `category_mapping.py` was needed or vendored.

## Not vendored (explicitly out of scope)

- `bfcl_eval/constants/model_config.py` (`MODEL_CONFIG_MAPPING`) — the model
  registry; see deviation (2) above.
- `bfcl_eval/model_handler/**` model handler implementations, and the
  `convert_to_tool` / `convert_to_function_call` tool-schema helpers.
- `bfcl_eval/eval_checker/multi_turn_eval/**`, `agentic_eval/**`, and the
  executable/REST checker — only the AST checker and relevance-detection
  helpers (`is_function_calling_format_output`, `is_empty_output`) are
  vendored.
- Any lighteval task/dataset wiring — that is the responsibility of a
  separate engineer working outside `_vendor_bfcl/`.
