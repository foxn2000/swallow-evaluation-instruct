# swallow-evaluation-instruct:
# This file is NOT a verbatim copy of a single upstream file. It is a hand-extracted
# subset of `bfcl_eval/model_handler/utils.py` and `bfcl_eval/utils.py`, containing
# only the functions transitively needed by four public entry points:
#   - ast_parse
#   - default_decode_ast_prompting
#   - formulate_system_prompt
#   - _func_doc_language_specific_pre_processing
# plus `is_function_calling_format_output` and `is_empty_output`, which upstream are
# defined in `bfcl_eval/utils.py` and re-exported (via `from bfcl_eval.utils import *`)
# through `bfcl_eval/eval_checker/eval_runner_helper.py`.
#
# Every function body below is copied verbatim from its upstream source (see the
# per-function comment for which file it came from); only import statements were
# rewritten to be relative to this vendored package. No logic was changed.
#
# Functions intentionally NOT vendored (out of scope for the four entry points above):
#   - convert_to_tool, convert_to_function_call, convert_value (tool-schema / execution
#     helpers, not needed for prompting-mode decode+AST-check)
#   - system_prompt_pre_processing_chat_model, convert_system_prompt_into_user_prompt,
#     combine_consecutive_user_prompts, extract_system_prompt, extract_last_user_message
#     (chat-message-list plumbing; the lighteval task builds its own prompt list)
#   - default_decode_execute_prompting, parse_nested_value, decoded_output_to_execution_list
#     (executable-category decoding, out of scope: AST checker only)
#   - retry_with_backoff, add_memory_instruction_system_prompt,
#     format_execution_results_prompting (agentic/memory/multi-turn helpers, out of scope)
#   - extract_test_category_from_id, extract_prompt_format_from_id, and the other
#     is_*() category predicates in bfcl_eval/utils.py that are not on the call graph
#     of the four entry points (is_java/is_js ARE needed and are included below)
#   - The big constant tables in bfcl_eval/constants/category_mapping.py: none of the
#     vendored functions turned out to depend on them (is_java/is_js/extract_test_category_from_id
#     upstream are plain substring/string-split checks, not table lookups), so nothing
#     from category_mapping.py was vendored.

import ast
import json
import re
from typing import Optional

# swallow-evaluation-instruct: rewrote absolute bfcl_eval imports to relative vendored imports.
from .default_prompts import (
    OUTPUT_FORMAT_MAPPING,
    PARAM_TYPE_MAPPING,
    PROMPT_STYLE_TEMPLATES,
    PROMPT_TEMPLATE_MAPPING,
)
from .enums import ReturnFormat
from .java_parser import parse_java_function_call
from .js_parser import parse_javascript_function_call
from .json_parser import parse_json_function_call
from .xml_parser import parse_concise_xml_function_call, parse_verbose_xml_function_call


#### From bfcl_eval/model_handler/utils.py ####


def ast_parse(
    input_str: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    if has_tool_call_tag:
        match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", input_str, re.DOTALL)
        if match:
            input_str = match.group(1).strip()
        else:
            raise ValueError(f"No tool call tag found in input string: {input_str}")

    if language == ReturnFormat.PYTHON:
        # We only want to remove wrapping quotes that could have been added by the model.
        cleaned_input = input_str.strip().strip("'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted

    elif language == ReturnFormat.JAVA:
        # Remove the [ and ] from the string
        # Note: This is due to legacy reasons, we should fix this in the future.
        return parse_java_function_call(input_str[1:-1])

    elif language == ReturnFormat.JAVASCRIPT:
        # Note: Same as above, we should fix this in the future.
        return parse_javascript_function_call(input_str[1:-1])

    elif language == ReturnFormat.VERBOSE_XML:
        # Remove ```xml and anything before/after XML
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(
                f"No XML function call found in input string: {input_str}. Missing <functions> tag."
            )
        return parse_verbose_xml_function_call(match.group(0))

    elif language == ReturnFormat.CONCISE_XML:
        # Remove anything before/after <functions> and </functions>
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(
                f"No XML function call found in input string: {input_str}. Missing <functions> tag."
            )
        return parse_concise_xml_function_call(match.group(0))

    elif language == ReturnFormat.JSON:
        json_match = re.search(r"\[.*\]", input_str, re.DOTALL)
        if json_match:
            input_str = json_match.group(0)
        return parse_json_function_call(input_str)

    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def default_decode_ast_prompting(
    result: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result, language, has_tool_call_tag)
    return decoded_output


def formulate_system_prompt(format_sensitivity_config: str, functions: list[dict]) -> str:
    """
    Formulate the default system prompt based on the provided parameters.
    """
    (
        return_format,
        has_tool_call_tag,
        function_doc_format,
        prompt_format,
        prompt_style,
    ) = parse_prompt_variation_params(format_sensitivity_config)

    formatted_function_doc = format_function_doc(functions, function_doc_format)

    prompt_template = PROMPT_TEMPLATE_MAPPING[prompt_format]
    style_template = PROMPT_STYLE_TEMPLATES[prompt_style]

    persona = style_template["persona"]
    task = style_template["task"]
    if has_tool_call_tag:
        tool_call_format = style_template["tool_call_with_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    else:
        tool_call_format = style_template["tool_call_no_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    multiturn_behavior = style_template["multiturn_behavior"]
    available_tools = style_template["available_tools"].format(
        format=function_doc_format,
        functions=formatted_function_doc,
    )

    system_prompt = prompt_template.format(
        persona=persona,
        task=task,
        tool_call_format=tool_call_format,
        multiturn_behavior=multiturn_behavior,
        available_tools=available_tools,
    )

    return system_prompt


def format_function_doc(functions: list[dict], function_doc_format: str) -> str:
    """
    Format the function documentation based on the specified format.
    """

    if function_doc_format == "xml":
        functions = _generate_function_doc_xml(functions)

    elif function_doc_format == "python":
        functions = _generate_function_doc_python(functions)

    elif function_doc_format == "json":
        functions = json.dumps(functions, indent=4)

    else:
        raise ValueError(f"Invalid function doc format: {function_doc_format}")

    return functions


def _generate_function_doc_xml(functions: list[dict]) -> str:
    """
    Generate the function documentation in XML format.
    """

    def _param_xml(
        name: str, meta: dict, required_set: Optional[list[str]], indent_lvl: int = 2
    ) -> str:
        """Recursively render a param and its nested structure to XML."""
        indent = " " * indent_lvl * 2  # 2 spaces per logical indent level

        p_type = meta.get("type", "string")
        p_desc = meta.get("description", "")
        # If there is no required set, then all parameters are required by default.
        if required_set is None:
            is_required = "true"
        else:
            is_required = "true" if name in required_set else "false"

        # Handle enum values
        if "enum" in meta:
            p_desc += f" Enum values: {meta['enum']}."

        # Handling for array/tuple/list types
        if "items" in meta and "type" in meta["items"]:
            inner_type = meta["items"]["type"]
            p_type = f"{p_type}[{inner_type}]"

        elif "additionalProperties" in meta:
            inner_type = meta["additionalProperties"].get("type", "string")
            p_type = f"{p_type}[{inner_type}]"

        # Build opening tag (include default attr if exists)
        attrs = [f'name="{name}", type="{p_type}", required="{is_required}"']
        if "default" in meta:
            attrs.append(f'default="{repr(meta["default"])}"')
        open_tag = f"{indent}<param " + " ".join(attrs).replace(",", "") + ">\n"

        xml_parts = [open_tag]
        xml_parts.append(f"{indent}  <desc>{p_desc}</desc>\n")

        # Recursive handling for object/dict types
        if "properties" in meta:
            child_required = meta.get("required", None)
            xml_parts.append(f"{indent}  <params>\n")
            for child_name, child_meta in meta["properties"].items():
                xml_parts.append(
                    _param_xml(child_name, child_meta, child_required, indent_lvl + 2)
                )
            xml_parts.append(f"{indent}  </params>\n")

        # closing tag
        xml_parts.append(f"{indent}</param>\n")
        return "".join(xml_parts)

    xml_blocks: list[str] = []
    for fn in functions:
        name = fn["name"]
        desc = fn.get("description", "")

        params_schema = fn["parameters"]
        top_props = params_schema.get("properties", {})
        top_required = params_schema.get("required", None)

        xml = f'<function name="{name}">\n'
        xml += f"  <desc>{desc}</desc>\n"
        xml += f"  <params>\n"

        for param_name, meta in top_props.items():
            xml += _param_xml(param_name, meta, top_required, indent_lvl=2)

        xml += f"  </params>\n"
        xml += f"</function>\n"
        xml_blocks.append(xml)

    return "\n".join(xml_blocks)


def _generate_function_doc_python(functions: list[dict]) -> str:
    """
    Generate the function documentation in Pythonic format.
    """

    def _to_py_type(meta: dict) -> str:
        t = meta.get("type", "string")
        primitive_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
        }

        if t in primitive_map:
            return primitive_map[t]

        if t in {"array", "list", "tuple"} and "items" in meta:
            return f"list[{_to_py_type(meta['items'])}]"

        if t in {"object", "dict"}:
            if "additionalProperties" in meta:
                return f"dict[str, {_to_py_type(meta['additionalProperties'])}]"
            # If specific properties, treat as dict
            return "dict"

        # Fallback
        return t

    INDENT_BASE = " " * 8  # 8 spaces inside the docstring block

    def _param_doc(name: str, meta: dict, depth: int = 0) -> list[str]:
        """Recursively build docstring lines for a parameter schema."""
        lines: list[str] = []
        indent = INDENT_BASE + (" " * 4 * depth)

        py_type = _to_py_type(meta)
        desc = meta.get("description", "")
        if "enum" in meta:
            desc += f" Enum values: {meta['enum']}."

        if "default" in meta:
            default_note = f", default={repr(meta['default'])}"
        else:
            default_note = ""
        lines.append(f"{indent}{name} ({py_type}{default_note}): {desc}\n")

        # Handle nested object properties
        if "properties" in meta:
            for child_name, child_meta in meta["properties"].items():
                lines.extend(_param_doc(f"{child_name}", child_meta, depth + 1))

        return lines

    docs: list[str] = []
    for fn in functions:
        full_name = fn["name"]
        desc = fn.get("description", "")

        doc_lines: list[str] = []
        doc_lines.append(f"# Function: {full_name}\n")
        doc_lines.append('    """\n')
        doc_lines.append(f"    {desc}\n\n")

        params_schema = fn.get("parameters", {})
        top_props = params_schema.get("properties", {})

        if top_props:
            doc_lines.append("    Args:\n")
            for param_name, meta in top_props.items():
                doc_lines.extend(_param_doc(param_name, meta))

        doc_lines.append('    """\n')
        docs.append("".join(doc_lines))
        docs.append("\n")

    return "\n\n".join(docs)


def parse_prompt_variation_params(input_str: str) -> tuple[str, bool, str, str, str]:
    """
    Parse a query string of the form:
      ret_fmt=…&tool_call_tag=…&func_doc_fmt=…&prompt_fmt=…&style=…

    Returns a 5-tuple containing, **in order**:
        1. return_format (str)
        2. has_tool_call_tag (bool)
        3. function_doc_format (str)
        4. prompt_format (str)
        5. prompt_style (str)

    Raises:
        ValueError: If the input string does not conform to the expected format.
    """
    _PATTERN = re.compile(
        r"^"
        r"ret_fmt=(?P<return_format>python|json|verbose_xml|concise_xml)"
        r"&tool_call_tag=(?P<has_tool_call_tag>True|False)"
        r"&func_doc_fmt=(?P<function_doc_format>python|xml|json)"
        r"&prompt_fmt=(?P<prompt_format>plaintext|markdown)"
        r"&style=(?P<prompt_style>classic|experimental)"
        r"$"
    )

    match = _PATTERN.match(input_str)
    if not match:
        raise ValueError(f"Invalid query format: {input_str!r}")

    # Extract named groups
    return_format = match.group("return_format")
    has_tool_call_tag = match.group("has_tool_call_tag") == "True"
    function_doc_format = match.group("function_doc_format")
    prompt_format = match.group("prompt_format")
    prompt_style = match.group("prompt_style")

    return (
        return_format,
        has_tool_call_tag,
        function_doc_format,
        prompt_format,
        prompt_style,
    )


#### From bfcl_eval/utils.py ####


def is_java(test_category):
    return "java" in test_category and not is_js(test_category)


def is_js(test_category):
    return "javascript" in test_category


def is_function_calling_format_output(decoded_output):
    """
    Ensure the output is a list of dictionaries of the form:
    `[{func1: {param1: val1, param2: val2, ...}}, {func2: {param1: val1, param2: val2, ...}}, ...]`
    Sometimes the model handler's `decode_ast` method will return successfully, but the output is not in the correct format, and that will mess up the downstream evaluation that expects this format.
    This is especially the case when the model doesn't predict any function calls, and the output is an human-readable string.
    Note: Empty list `[]` is considered the correct format in this check.
    """
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        # Check for `{func1: {param1: val1, param2: val2, ...}}`, should only have one key-value pair
        if len(item) != 1:
            return False
        # Check for `{param1: val1, param2: val2, ...}`; the parameter-value pairs should be a dictionary
        if type(list(item.values())[0]) != dict:
            return False
    return True


def is_empty_output(decoded_output):
    # This function is a patch to the ast decoder for relevance detection
    # Sometimes the ast decoder will parse successfully, but the input doens't really have a function call
    # [], [{}], and anything that is not in function calling format is considered empty (and thus should be marked as correct)
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


def _get_language_specific_hint(test_category):
    if is_java(test_category):
        return " Note that the provided function is in Java 8 SDK syntax."
    elif is_js(test_category):
        return " Note that the provided function is in JavaScript syntax."
    else:
        return " Note that the provided function is in Python 3 syntax."


def _func_doc_language_specific_pre_processing(
    function: list[dict], test_category: str
) -> list[dict]:
    if len(function) == 0:
        return function

    assert type(function) == list
    for item in function:
        # Add language specific hints to the function description
        item["description"] = item["description"] + _get_language_specific_hint(
            test_category
        )
        # Process the parameters
        properties = item["parameters"]["properties"]
        if is_java(test_category):
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of Java object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is Java {value['type']} type parameter in string representation."
                if value["type"] == "ArrayList" or value["type"] == "Array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                value["type"] = "string"

        elif is_js(test_category):
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key][
                        "description"
                    ] += " This parameter can be of any type of JavaScript object in string representation."
                else:
                    value[
                        "description"
                    ] += f" This is JavaScript {value['type']} type parameter in string representation."
                if value["type"] == "array":
                    value[
                        "description"
                    ] += f" The list elements are of type {value['items']['type']}; they are not in string representation."
                    del value["items"]

                if value["type"] == "dict":
                    if "properties" in value:  # not every dict has properties
                        value[
                            "description"
                        ] += f" The dictionary entries have the following schema; they are not in string representation. {json.dumps(value['properties'])}"
                        del value["properties"]

                value["type"] = "string"

    return function
