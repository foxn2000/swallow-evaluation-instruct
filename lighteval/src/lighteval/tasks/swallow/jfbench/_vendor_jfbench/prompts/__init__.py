from .ifbench import get_all_ifbench_prompts
from .ifbench import IFBenchPrompt


# swallow-evaluation-instruct: ja_stackoverflow prompt source (and its module) is not vendored;
# this package only re-exports the IFBench prompt loader.
__all__ = [
    "IFBenchPrompt",
    "get_all_ifbench_prompts",
]
