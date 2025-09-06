from dataclasses import dataclass
from typing import Optional
import re
from collections import defaultdict


def extract_until(s1, s2):
    idx = s1.find(s2)
    return s1[:idx] if idx != -1 else s1


def get_tags(tag, format='bespoke'):
    if format == 'bespoke':
        start = f"<|begin_of_{tag}|>"
        end = f"<|end_of_{tag}|>"
    elif format == 'r1':
        start = f"<{tag}>"
        end = f"</{tag}>"
    else:
        raise NotImplementedError
    return start, end


def grab_text_between(s, start, end):

    pattern = re.escape(start) + r"(.*?)" + re.escape(end)

    matches = re.findall(pattern, s, re.DOTALL)
    if matches is None:
        print(f'Unable to find content in {start} {end}')
        raise ValueError
    return matches[0]


def grab_text_between_tag(s, tag, tag2=None, format='bespoke'):

    start, end = get_tags(tag, format=format)

    if tag2 is not None:
        start = end
        end, _ = get_tags(tag2, format=format)
    return grab_text_between(s, start, end)


def wrap_string_between_tag(s: str, tag: str, format='bespoke'):
    start, end = get_tags(tag=tag, format=format)
    if s.startswith(start):
        assert s.endswith(end)
        return s
    return f"{start}{s}{end}"


@dataclass
class ReasoningData:
    system_prompt: str
    think_tag: str
    solution_tag: str

    think_prefix: Optional[str]
    think_solution_delimiter: Optional[str]
    delimiters_padding: Optional[str]
    tag_format: str = 'bespoke'

    def extract_question_and_completion_from_line(self, line):

        user_message = line["conversations"][0]
        assert user_message['from'] == 'user'
        question_content = user_message["value"]
        assistant_message = line["conversations"][1]
        assert assistant_message['from'] == 'assistant'
        thought_process_and_solution = assistant_message["value"]
        return question_content, thought_process_and_solution

    def get_tags(self,):
        start_think_tag, end_think_tag = get_tags(
            tag=self.think_tag, format=self.tag_format)
        start_solution_tag, end_solution_tag = get_tags(
            tag=self.solution_tag, format=self.tag_format)
        return (start_think_tag, end_think_tag,
                start_solution_tag, end_solution_tag)

    def format_reasoning_message(
            self,
            think_content,
            solution_content,
            think_prefix=None,
            think_solution_delimiter=None,
    ):
        if self.think_prefix is not None:
            think_prefix = self.think_prefix
        if think_prefix is None:
            think_prefix = ''
        if self.think_solution_delimiter is not None:
            think_solution_delimiter = self.think_solution_delimiter
        if think_solution_delimiter is None:
            think_solution_delimiter = '\n\n'
        think_s = wrap_string_between_tag(
            s=think_content, tag=self.think_tag, format=self.tag_format)

        solution_s = wrap_string_between_tag(
            s=solution_content, tag=self.solution_tag, format=self.tag_format)
        return think_prefix, think_s, think_solution_delimiter, solution_s

    def extract_reasoning_message_content(self, s):
        if self.think_prefix is not None:
            think_prefix = self.think_prefix
        else:
            think_prefix = ''
        if self.think_solution_delimiter is not None:
            think_solution_delimiter = self.think_solution_delimiter
        else:
            think_solution_delimiter = grab_text_between_tag(
                s=s, tag=self.think_tag, tag2=self.solution_tag,
                format=self.tag_format)

        solution_content = grab_text_between_tag(
            s=s, tag=self.solution_tag, format=self.tag_format)
        think_content = grab_text_between_tag(
            s=s, tag=self.think_tag, format=self.tag_format)
        return (think_prefix, think_content, think_solution_delimiter,
                solution_content)


S1_QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()


@dataclass
class CustomReasoningData(ReasoningData):
    def extract_question_and_completion_from_line(self, line):
        question_content = line["prompt"]
        solution = line["solution"]
        # if unavailable defaults to the solution
        thought_process = line.get("reasoning_trace", solution)
        thought_process_and_solution = wrap_string_between_tag(
            s=thought_process, tag=self.think_tag) + wrap_string_between_tag(
                s=solution, tag=self.solution_tag)
        return question_content, thought_process_and_solution


STRATOS_SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
)


STRATOS_CONFIG = ReasoningData(
    system_prompt=STRATOS_SYSTEM_PROMPT,
    think_tag="thought",
    solution_tag="solution",
    think_prefix="",
    think_solution_delimiter="\n\n",
    delimiters_padding="\n\n",
)


CUSTOM_CONFIG_STRATOS_STYLE = CustomReasoningData(
    system_prompt=STRATOS_SYSTEM_PROMPT,
    think_tag="thought",
    solution_tag="solution",
    think_prefix="",
    think_solution_delimiter="\n\n",
    delimiters_padding="\n\n"
)


DATA_CONFIGS = defaultdict(
    lambda: CUSTOM_CONFIG_STRATOS_STYLE,
    {'bespokelabs/Bespoke-Stratos-17k': STRATOS_CONFIG}
)
