import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedAnswer:
    reasoning: str
    answer: str | None
    valid: bool
    error: str | None = None


def parse_answer(raw, choices=None):
    thoughts = re.findall(r"<THINK>(.*?)</THINK>", raw, re.I | re.S)
    reasoning = "\n".join(t.strip() for t in thoughts)
    answers = re.findall(r"<ANSWER>(.*?)</ANSWER>", raw, re.I | re.S)
    if len(answers) != 1 or len(re.findall(r"<ANSWER>", raw, re.I)) != 1:
        return ParsedAnswer(reasoning, None, False, "Expected exactly one complete ANSWER tag")
    answer = answers[0].strip()
    if not answer or "<" in answer or (choices and answer not in choices):
        return ParsedAnswer(reasoning, None, False, "Invalid answer payload")
    return ParsedAnswer(reasoning, answer, True)
