import json

from lighteval.tasks.swallow.jfbench._vendor_jfbench._data import DATA_DIR
from lighteval.tasks.swallow.jfbench._vendor_jfbench.protocol import Constraint


DATA_PATH = str(DATA_DIR / "ifbench_ja_translated.jsonl")
JA_PROMPT_COL = "japanese_prompt_without_constraints"


class IFBenchPrompt:
    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        constraints_instructions = "\n".join(
            f"- {constraint.instructions(train_or_test=train_or_test)}"
            for constraint in constraints
        )
        return (
            "# 指示文\n"
            f"{self._prompt}\n\n"
            "# 回答に関する注意事項\n"
            "特に指定がなければ、日本語で回答してください。ただし、英語での回答が求められている場合は英語で回答してください。\n"
            "ただし、以下の制約条件を全て守ってください。\n"
            f"{constraints_instructions}"
        )

    @property
    def document(self) -> str:
        return self._prompt


def get_all_ifbench_prompts(dataset_path: str | None = None) -> list[IFBenchPrompt]:
    path = dataset_path or DATA_PATH
    # swallow-evaluation-instruct: upstream used pandas.read_json(orient="records", lines=True)
    # to load the JSONL and pull out the JA_PROMPT_COL column; replaced with plain json/open()
    # line-by-line parsing so pandas is not a dependency of the vendored package.
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(record[JA_PROMPT_COL])
    return [IFBenchPrompt(prompt) for prompt in prompts]
