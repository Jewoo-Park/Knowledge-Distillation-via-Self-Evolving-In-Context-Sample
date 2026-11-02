"""Student input: the vanilla prompt, optionally preceded by retrieved in-context samples (text only, no example video).

Shown examples have their options relabelled at random (fixed per question and item), with the answer letter and
letter mentions following the option text, so the letter of a similar past question carries no information.
"""
import random
import re
from src.utils.storage import digest

VIDEO = '<VIDEO>'
FORMAT_TEXT = ('Output format:\nFirst describe the relevant visible events and briefly explain your answer inside '
               '<THINK> and </THINK>. Then give the final answer inside <ANSWER> and </ANSWER>. Include both blocks, in '
               'that order:\n<THINK>Brief evidence-based reasoning</THINK>\n<ANSWER>exactly one option letter</ANSWER>')
EXAMPLES_HEADER = 'Solved examples from other videos, for reference:\n'
EXAMPLES_FOOTER = 'Now answer the new question about the following video.\n\n'
LETTER_REF = re.compile(r'(\b(?:option|choice)\s*\(?)([A-E])\b|\(([A-E])\)|\b([A-E])(?=\.)', re.I)


def question_text(question, choices):
    return 'Question:\n' + question + '\nChoices:\n' + '\n'.join(f'{k}. {v}' for k, v in choices.items())


def user_prompt(sample):
    """Byte-identical to the vanilla run's user prompt."""
    return question_text(sample.question, sample.choices) + '\n\n' + FORMAT_TEXT


def clean_output(raw):
    """THINK + ANSWER blocks only; None when either is missing."""
    think = re.findall(r'<THINK>(.*?)</THINK>', raw, re.S | re.I)
    answer = re.findall(r'<ANSWER>(.*?)</ANSWER>', raw, re.S | re.I)
    if not think or len(answer) != 1:
        return None
    return f'<THINK>{" ".join(t.strip() for t in think)}</THINK>\n<ANSWER>{answer[0].strip()}</ANSWER>'


def permute_options(choices, example, rng):
    """(choices, example) with option texts moved to random letters; ANSWER and letter mentions follow the texts."""
    letters = list(choices)
    moved = letters[:]
    rng.shuffle(moved)
    m = dict(zip(letters, moved))
    new_choices = {m[a]: choices[a] for a in letters}
    new_choices = {a: new_choices[a] for a in sorted(new_choices)}

    def sub(t):
        if t.group(2):
            return t.group(1) + m.get(t.group(2).upper(), t.group(2))
        if t.group(3):
            return '(' + m.get(t.group(3), t.group(3)) + ')'
        return m.get(t.group(4), t.group(4))
    head, letter, tail = re.match(r'(.*)<ANSWER>(.*?)</ANSWER>(.*)', example, re.S).groups()
    return new_choices, LETTER_REF.sub(sub, head) + '<ANSWER>' + m[letter.strip()] + '</ANSWER>' + tail


def example_block(sample_id, items, permute=True):
    out = [EXAMPLES_HEADER]
    for i, item in enumerate(items, 1):
        choices, example = item['choices'], item['example']
        if permute:
            choices, example = permute_options(choices, example, random.Random(digest({'s': sample_id, 'item': item['id']})))
        out.append(f'Example {i}\n{question_text(item["question"], choices)}\nAnswer:\n{example}\n')
    out.append(EXAMPLES_FOOTER)
    return '\n'.join(out)


def student_parts(sample, items, permute=True):
    """None (= vanilla input) without items, else [examples, VIDEO, question + format]."""
    if not items:
        return None
    return [example_block(sample.sample_id, items, permute), VIDEO, user_prompt(sample)]
