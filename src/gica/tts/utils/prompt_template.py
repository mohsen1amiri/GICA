"""
prompt_template.py
==================

Prompt formatting for the reasoning-based PRM verifier (ThinkPRM).

:func:`format_verification_cot_for_thinkprm` is the inference template used in the paper:
it wraps the question and a (possibly partial) step-by-step solution into the chat prompt
ThinkPRM was trained on, asking it to review and critique each step. The default
instruction matches the one used during training and can be overridden via ``instruction``.

The remaining two functions are kept for completeness: a training-time formatter and a
template for non-thinking verifiers (the latter is not used in the paper).

Note:
    The exact wording and whitespace (including the blank lines) of the templates are
    significant -- they reproduce the format ThinkPRM was trained with -- and must not be
    altered.
"""


def format_verification_cot_for_thinkprm(tokenizer, problem, solution, cot=None, long_cot=False, instruction=None):
    """Build the ThinkPRM inference prompt for verifying ``solution`` to ``problem``.

    Applies the model's chat template with a generation prompt appended. ``instruction``
    overrides the default critique instruction (which is the one the model was trained
    with). ``cot`` and ``long_cot`` are accepted for interface compatibility.
    """
    # Default instruction; the verifier models were trained with this exact wording.
    _instruction = instruction if instruction is not None else "Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps."

    instruction_template = """You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{solution}

{_instruction}
""".strip()

    s = tokenizer.apply_chat_template([
        {'role': "user", "content": instruction_template.replace('{problem}', problem).replace('{solution}', solution).replace('{_instruction}', _instruction)}
    ], tokenize=False, add_generation_prompt=True)

    return s


def format_train_verification_cot_for_thinkprm(tokenizer, problem, solution, cot=None):
    """Format a verification example for ThinkPRM, for either training or inference.

    When ``cot`` is provided (training), the verification trace is spliced into an
    assistant turn: the text after ``</think>`` is preserved, a "Let's verify step by
    step:" lead-in is injected into the ``<think>`` block if absent, and the EOS token is
    re-appended. When ``cot`` is ``None`` (inference), only the user turn is emitted with a
    generation prompt and the same "Let's verify step by step:" lead-in.
    """
    instruction = """You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{problem}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps.
""".strip()
    if cot:  # training
        # Preserve the content after </think>, then strip </think> from the trace.
        after_think = cot[cot.index('</think>')+len('</think>'):]
        cot = cot[:cot.index('</think>')]

        # Ensure the trace opens with the standard "verify step by step" lead-in.
        if "Let's verify step by step:" not in cot:
            cot = cot.replace("<think>", "<think>\nLet's verify step by step:")

        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction.replace('{problem}', problem).replace('{solution}', solution)},
            {'role': "assistant", "content": f"{cot}"}
        ], tokenize=False, add_generation_prompt=False).replace(tokenizer.eos_token, '')

        # Re-attach the post-</think> content and a terminating EOS.
        s += '</think>' + after_think
        s += tokenizer.eos_token

    else:  # inference
        s = tokenizer.apply_chat_template([
            {'role': "user", "content": instruction.replace('{problem}', problem).replace('{solution}', solution)},
        ], tokenize=False, add_generation_prompt=True) + "\nLet's verify step by step:"

    return s


def format_verification_cot_no_think(tokenizer, problem, solution, cot=None):
    """Format a verification prompt for a non-thinking (direct-judgment) verifier.

    Not used in the paper. Falls back to a plain text prompt if the tokenizer has no chat
    template.
    """
    instruction = ("Given a math question and partial solution steps, analyze each step in the solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct.")
    try:
        if cot:
            return tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                {'role': "assistant", "content": f"Analysis:\n{cot}"}
            ], tokenize=False)
        else:
            s = tokenizer.apply_chat_template([
                {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                {'role': "assistant", "content": ""}
            ], tokenize=False, add_generation_prompt=False)
            return s.replace(tokenizer.eos_token, '')
    except Exception:
        if cot:
            return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\nAnalysis:\n{cot}{tokenizer.eos_token}"
        else:
            return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\n"
