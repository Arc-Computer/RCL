from datasets import load_dataset
import re
from typing import Optional
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData, extract_until)

STRATOS_STYLE_TEACHER_PROMPT = str(
    "Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. "
    "Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Solution and Explanation. "
    "In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. "
    "In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\\n\\n'} <|end_of_explanation|> Each step should show detailed considerations leading to your solutions such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
)

TEACHER_SYSTEM_PROMPTS_AND_TAGS = {
    'stratos': (
        STRATOS_STYLE_TEACHER_PROMPT,  'solution', 'explanation', '',
        '\n\n', '\n\n'),
}


def process_assistant_message(
    solution_content, think_content, new_solution_tag, explanation_tag
):
    solution = wrap_string_between_tag(
        s=solution_content, tag=new_solution_tag)
    explanation = wrap_string_between_tag(s=think_content, tag=explanation_tag)
    return solution, explanation


def get_teacher_data_mask_delimiter(
        mask_teacher_answer,
        system_prompt_style='stratos',
        include_end_of_solution_tag=False,
        dataset_id_or_path=None,
):
    custom_mask_delimiter = None
    if mask_teacher_answer:
        _, _solution_tag, explanation_tag, _, delimiter, paddings = (
            TEACHER_SYSTEM_PROMPTS_AND_TAGS[system_prompt_style])
        if dataset_id_or_path is not None:
            data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
            paddings = data.delimiters_padding
        start_expl_tag, _ = get_tags(tag=explanation_tag)
        custom_mask_delimiter = start_expl_tag + paddings
        if include_end_of_solution_tag:
            _, end_soln_tag = get_tags(tag=_solution_tag)
            custom_mask_delimiter = (
                end_soln_tag + delimiter + custom_mask_delimiter)
    else:
        raise NotImplementedError
    return custom_mask_delimiter


def get_process_line_fn(
        dataset_id_or_path,
        system_prompt_style='stratos',
        return_student_prompt_info: bool = True,
        add_text_completions: bool = False):
    # Data formatting for teacher RL and SFT
    print(f'Creating data for format {dataset_id_or_path}')
    data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
    (system_msg, new_solution_tag, explanation_tag, new_pre, new_delimiter,
     new_paddings) = (TEACHER_SYSTEM_PROMPTS_AND_TAGS[system_prompt_style])
    new_paddings = data.delimiters_padding

    def process_line_fn(line, tokenizer):
        messages = [{
            "role": "system",
            "content": system_msg,
        },
        ]
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        messages.append(
            {"role": 'user', "content": question_content},)
        (think_prefix, think_content, think_solution_delimiter,
            solution_content) = data.extract_reasoning_message_content(
            s=thought_process_and_solution)
        solution, explanation = process_assistant_message(
            solution_content=solution_content,
            think_content=think_content,
            new_solution_tag=new_solution_tag,
            explanation_tag=explanation_tag,
        )
        new_content = new_pre + solution + new_delimiter + explanation
        messages.append(
            {"role": 'assistant', "content": new_content},)

        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        out_line = {'text': line_text}
        if add_text_completions:
            assert return_student_prompt_info
        if return_student_prompt_info:
            start_explanation_tag, end_explanation_tag = get_tags(
                tag=explanation_tag)
            start_solution_tag, end_solution_tag = get_tags(
                tag=new_solution_tag)
            completion_div = (
                end_solution_tag + new_delimiter + start_explanation_tag +
                new_paddings)

            rl_prompts, rl_solutions = line_text.split(completion_div)
            (start_think_student_tag, end_think_student_tag,
             start_solution_student_tag, end_solution_student_tag
             ) = data.get_tags()

            solution_s = wrap_string_between_tag(
                s=solution_content, tag=data.solution_tag)

            rl_out_line = {
                "prompt": rl_prompts + completion_div,
                # information for the reward function
                "student_system_prompts": data.system_prompt,
                "start_think_teacher_tags": start_explanation_tag,
                "end_think_teacher_tags": end_explanation_tag,
                "start_think_student_tags": start_think_student_tag,
                "end_think_student_tags": end_think_student_tag,
                "start_solution_tags": start_solution_student_tag,
                "end_solution_tags": end_solution_student_tag,
                "think_prefixes": think_prefix,
                "think_solution_delimiters": think_solution_delimiter,
                "questions": question_content,
                "solutions": solution_s,
                "solutions_content": solution_content,
                "delimiters_padding": new_paddings,
                "masked_out_think_prefix": start_explanation_tag + new_paddings,
            }
            if add_text_completions:
                rl_out_line['completion'] = rl_solutions
            out_line.update(rl_out_line)
        return out_line
    return process_line_fn
