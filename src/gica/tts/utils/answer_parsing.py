"""
answer_parsing.py
=================

Answer-parsing utilities for the TTS pipeline.

The one routine the verifier depends on is :func:`extract_step_labels`, which converts a
ThinkPRM verification trace into a list of per-step binary correctness labels by reading
its boxed decision tokens. The remaining helpers (GSM8k-style answer retrieval, exact/
numeric answer judging, majority voting, and plan action-trace extraction) are inherited
from the upstream GenPRM pipeline and are kept for completeness.
"""

import re
from typing import Optional
from collections import Counter
import re


def retrieve_answer(output: str) -> Optional[str]:
    """Extract the final answer from a model output of the form ``"The answer is ..."``.

    Strips commas, dollar signs, and spaces, and (if the answer contains ``=``) keeps the
    text after the last ``=``. Returns ``None`` when no answer phrase is found.
    """
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    """Extract the gold answer following the ``####`` delimiter (GSM8k convention)."""
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    """Return whether ``output`` matches ``answer``.

    Compares as integers first, then as floats, then falls back to a string comparison.
    A ``None`` output is always judged incorrect.
    """
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except Exception as e:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except Exception as e:
        pass
    return output == answer


def get_majoirty_answer(answers):
    """Return the most frequent answer (self-consistency vote).

    Returns ``'[INVALID]'`` if ``answers`` is empty (i.e. no candidate produced a final
    answer); otherwise the single most common entry.
    """
    if len(answers) == 0:  # all solutions are invalid -- none produced a final answer
        return '[INVALID]'
    else:
        # Most frequent answer across the candidates.
        voted_answer = Counter(answers).most_common(1)[0][0]
        return voted_answer


def get_action_trace_from_plan_str(plan_str):
    """Return the lines that immediately follow each ``[action]`` marker, joined by newlines."""
    lines = plan_str.split('\n')
    action_trace = []
    for i, line in enumerate(lines):
        if '[action]' in line and i + 1 < len(lines):
            action_trace.append(lines[i+1])

    return "\n".join(action_trace)


def extract_step_labels(output_str, correct_token, incorrect_token):
    """Parse per-step binary correctness labels from a ThinkPRM verification trace.

    Scans for boxed decision tokens ``\\boxed{<correct_token>}`` / ``\\boxed{<incorrect_token>}``
    in order and maps them to ``1`` (correct) and ``0`` (incorrect) respectively.

    Args:
        output_str (str): The verifier's generated text.
        correct_token (str): Token denoting a correct step (e.g. ``"+"``).
        incorrect_token (str): Token denoting an incorrect step (e.g. ``"-"``).

    Returns:
        list[int]: One label per boxed decision token, in order of appearance.
    """
    step_labels = []
    # Match boxed decision tokens (either the correct or the incorrect marker).
    pattern = r'\\boxed\{(' + re.escape(correct_token) + '|' + re.escape(incorrect_token) + r')\}'
    matches = re.findall(pattern, output_str)

    for match in matches:
        if match == correct_token:
            step_labels.append(1)
        elif match == incorrect_token:
            step_labels.append(0)

    return step_labels
