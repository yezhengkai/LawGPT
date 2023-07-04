"""
A dedicated helper to manage templates and prompt building.
"""
import os.path as osp

from lawgpt.utils import load_json


TEMPLATES_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "templates")


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(
        self,
        template_name: str = "",
        verbose: bool = False,
        templates_dir: str | None = None,
    ):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        if templates_dir is None:
            file_name = osp.join(TEMPLATES_DIR, f"{template_name}.json")
        else:
            file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        self.template = load_json(file_name)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: str | None = None,
        label: str | None = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
