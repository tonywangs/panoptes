"""Judges score (task, response) pairs and return RubricScore via structured output.

The Protocol is in `base.py`; rubric-style judges in `rubric.py`. Pairwise
judges (MT-Bench style) are a planned follow-up.
"""

from panoptes.judges.base import Judge, PromptTemplate, load_prompt_template
from panoptes.judges.rubric import RubricJudge

__all__ = ["Judge", "PromptTemplate", "RubricJudge", "load_prompt_template"]
