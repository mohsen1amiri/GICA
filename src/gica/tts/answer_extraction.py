"""
answer_extraction.py
====================

Final-answer extraction and string normalization used to score the TTS pipeline by
Exact-Match (EM). After GICA / a baseline selects a winning reasoning path, the path's
free-form text is reduced to a canonical answer string here, then compared against the
ground truth.

These routines are adapted from the math-evaluation utilities of Qwen2.5-Math and the
upstream GenPRM pipeline; ``strip_string`` performs the heavy LaTeX/format normalization
(fractions, roots, units, degrees, currency, infinity, etc.) so that two strings denoting
the same mathematical answer compare equal.

Note:
    ``extract_answer``'s "boxed" and program-output branches call
    ``extract_boxed_answers`` and ``extract_program_output``, which are expected to be
    available in the importing scope; they are not defined in this module. The TTS drivers
    import only ``strip_string``, so those branches are not exercised by the pipeline.
"""

import re
import regex


def extract_answer(pred_str, exhaust=False):
    """Extract the final answer string(s) from a model's free-form output.

    Resolution order: an explicit "final answer is $...$" template, a ``\\boxed{...}``
    answer, a "the answer is ..." phrase, then a program-output fallback, and finally the
    last number in the text. Each candidate is trimmed and normalized via ``strip_string``.

    Args:
        pred_str (str): Raw model output.
        exhaust (bool): If True, return the list of all extracted candidates; otherwise
            return the last (most specific) candidate, or "" if none was found.

    Returns:
        list[str] | str: All candidates when ``exhaust`` is True, else a single string.
    """
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else:  # use the last number
            pattern = r"-?\d*\.?\d+"
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if len(ans) >= 1:
                ans = ans[-1]
            else:
                ans = ""
            if ans:
                pred.append(ans)

    # Keep only the first line of each candidate and trim surrounding punctuation.
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def strip_string(string):
    """Normalize an answer string to a canonical form for Exact-Match comparison.

    Applies a long sequence of LaTeX/format rewrites so that visually different but
    mathematically equivalent answers map to the same string (e.g. ``\\dfrac`` -> ``frac``,
    ``\\sqrt2`` -> ``\\sqrt{2}``, stripped units/degrees/currency, normalized fractions and
    trailing zeros). Returns the normalized string.
    """
    string = str(string).strip()
    # Drop line breaks.
    string = string.replace("\n", "")

    # Trailing period.
    string = string.rstrip(".")

    # Remove inverse (negative) spacing.
    string = string.replace("\\!", "")

    # Unwrap a whole-string \text{...} wrapper.
    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # Canonicalize fraction macros to \frac.
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # Drop \left and \right delimiters.
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove a trailing \text{...} unit (e.g. miles, dollars) if it leaves a non-empty answer.
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    # Remove degree symbols.
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    # Remove trailing units: (c|m)m optionally squared/cubed, p.m., and a trailing "t".
    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # Remove dollar signs.
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # Drop a leading "x\in" qualifier.
    string = string.replace("x\\in", "")

    # Normalize escaped percent signs (percent itself is intentionally preserved).
    string = string.replace("\\%", "%")
    string = string.replace("\\%", "%")

    # Add a leading zero to bare decimals (" .5" -> " 0.5", "{.5" -> "{0.5").
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # Drop multiplication dots.
    string = string.replace("\\cdot", "")

    # Canonicalize infinity spellings.
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # Drop font macros.
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # Remove \mbox{...}.
    string = re.sub(r"\\mbox{.*?}", "", string)

    # (No-ops in the original: quote stripping is not assigned back; preserved as-is.)
    string.replace("'", "")
    string.replace('"', "")

    # Treat a lone imaginary unit "j" as "i".
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # Collapse "a.000b" -> "ab" and "a.000" -> "a" (strip redundant zero decimals).
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # Empty -> return as-is; leading "." -> prepend "0".
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Structural fixes: roots, tangents, whitespace, fractions, and a/b -> \frac.
    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # Trim trailing backslashes / commas / periods.
    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def _fix_fracs(string):
    """Rewrite shorthand fractions into ``\\frac{...}{...}`` (e.g. ``\\frac12`` -> ``\\frac{1}{2}``).

    Handles forms like ``\\frac1b``, ``\\frac12``, and ``\\frac1{72}``. Returns the input
    unchanged if a fragment cannot be parsed.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Rewrite a simple ``a/b`` into ``\\frac{a}{b}`` when both parts are integers (or sqrt).

    Returns the input unchanged if it is not a single ``a/b`` of the expected form.
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    """Brace bare square-root arguments: ``\\sqrt2`` / ``\\sqrt 2`` -> ``\\sqrt{2}``."""
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    """Brace bare tangent arguments: ``\\tan30`` / ``\\tan 30`` -> ``\\tan{30}``."""
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string
