"""Prompts, schemas and input rendering for the five-stage ICS evolution.

Each stage is its own call with only the information it needs:
  1 select   (critic, text)   block records -> cases worth a visual re-check
  2 solve    (teacher, frames) ONLY frames + question + options of the selected cases (see teacher_view)
  3 compare  (critic, text)   answers, student outputs, teacher solutions, related memory -> worked-example candidates
  4 refine   (critic, text)   candidates (+ student trial results) -> when_to_use, strategy, worked_output, next_action
  5 manage   (critic, text)   memory, candidates with trials, recent online usage -> add/refine/merge/prune/hold
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


# ------------------------------------------------------------------ stage 1
class SelectedCase(Strict):
    case_id: str
    priority: int
    selection_reason: str
    visual_questions_to_verify: list[str]


class Stage1Output(Strict):
    selected_cases: list[SelectedCase]


STAGE1_PROMPT = """You select, from a small frozen video-language model's solving records, the questions most worth a
visual re-check by a teacher who will watch the video again. The model is Qwen3.5-9B (32 frames of about 256x256
pixels); before some questions it was shown one solved example from its memory.

Priorities:
- observation or judgment errors that recur across several questions;
- cases where the answer without examples was right and became wrong with the example;
- cases where the output seems to copy the example's objects, numbers, directions or conclusion into this video;
- judgment types the current memory does not cover;
- correct answers whose reasoning may be wrong.

You have not seen the videos: do not state that a perception failure happened. For each selected case, state what
should be checked in the video to tell the possible causes apart. Do not pick only similar questions; weigh the value
of the re-check and the variety of question types.

Do not write examples and do not change the memory at this stage.
Select at most {max_selected} cases. Return JSON: selected_cases, each with case_id, priority (1 = highest),
selection_reason and visual_questions_to_verify."""


# ------------------------------------------------------------------ stage 2
class Evidence(Strict):
    frames: str
    observation: str
    basis: Literal['seen', 'inferred']


class TeacherSolution(Strict):
    case_id: str
    answer: str
    evidence: list[Evidence]
    answer_relation: str
    uncertainties: list[str]
    confidence: Literal['high', 'medium', 'low']


class Stage2Output(Strict):
    solutions: list[TeacherSolution]


STAGE2_PROMPT = """Solve each multiple-choice question independently, from the given video frames, question and options
only.

Explain the key observations that support your answer and why they lead to that option. Give each observation its
frame numbers or times. Mark whether each one is seen directly on screen or inferred, and do not make up details you
cannot verify.

Do not treat a route, goal or expected action written in the question as something the video already shows. When the
needed information is missing or the interpretation is uncertain, say so.

Return JSON: solutions, one per case_id, each with answer (one option letter), evidence (frames, observation, basis),
answer_relation, uncertainties and confidence (high / medium / low)."""


# ------------------------------------------------------------------ stage 3
class Choice(Strict):
    letter: str
    text: str


class CaseAnalysis(Strict):
    case_id: str
    verdict: Literal['student_wrong_teacher_supported', 'both_right_student_sound', 'both_right_student_unsound',
                     'teacher_wrong', 'teacher_insufficient', 'unclear']
    analysis: str


class Candidate(Strict):
    source_case_ids: list[str]
    kind: Literal['success', 'corrected']
    question: str
    choices: list[Choice]
    worked_output: str
    supported_facts: list[str]
    transferable_operation: str
    risks: str


class Stage3Output(Strict):
    failure_analysis: list[CaseAnalysis]
    candidates: list[Candidate]


STAGE3_PROMPT = """Compare the student's and the teacher's solutions of these questions and find observation and judgment
steps worth reusing on other videos. The student is a small frozen video-language model (Qwen3.5-9B); the teacher
solved each question independently from the same 32 frames without seeing the answer or the student's output.

Do not trust a teacher solution only because its answer is right. Check whether its observations are sufficient and
whether the link from observation to conclusion holds. If it is insufficient, record that further checking is needed
and do not invent evidence.

When the student was wrong and the teacher was right with evidence, you may write a candidate that corrects the missed
observation or the wrong judgment. When both were right, use the case as a success candidate only if the student's
reasoning is sound. Never use a case the teacher got wrong as the basis of a new correct example; record it as a
limit of the teacher or a counterexample to the existing memory.

A candidate must be a worked example built on one real past question: do not replace it with an abstract lesson
sentence, and do not merge observations from different videos into one scene that never existed. The example must
show the chain: concrete observation -> comparison, count or judgment -> answer. Correctness and usefulness come
before keeping the student's wording. Cut background description that is not needed, but keep the numbers, objects
and times the judgment depends on.

Return JSON: failure_analysis (one per case: case_id, verdict, analysis) and candidates, each with source_case_ids (the
first id is the question the example is about), kind (success / corrected), question, choices, worked_output,
supported_facts, transferable_operation and risks. Write worked_output in the student's answer format:
<THINK>...</THINK> then a new line and <ANSWER>letter</ANSWER>, answering the correct option of that question."""


# ------------------------------------------------------------------ stage 4
class CandidateRevision(Strict):
    candidate_id: str
    when_to_use: str
    strategy: str
    worked_output: str
    changes: str
    trial_review: str
    next_action: Literal['more_trial', 'revise', 'memory_review', 'hold', 'discard']


class Stage4Output(Strict):
    candidates: list[CandidateRevision]


STAGE4_PROMPT = """Your goal is not an example that looks general to a teacher, but one that Qwen3.5-9B (a small frozen
video-language model) can use on NEW videos. The model sees the example's question, options (relabelled at random,
with the letters inside the example changed to match) and worked_output as a solved example from another video,
placed {position}. It does not see when_to_use or strategy.

When writing a candidate:
- when_to_use: the question type, the kind of observation needed and the common trap. It is embedded and compared
  with new questions (video + question + options) to decide whether to show the example.
- Do not put shop names, person names, place names or exact floor numbers of the source video into the applicability
  condition unless needed; but keep the categories that separate judgments, such as camera versus object motion or
  repeated versus one-off actions.
- Do not keep narrowing the condition so that it only applies where one particular answer is right.
- strategy: one sentence with the key judgment the example shows.
- The worked_output itself must show that judgment. Avoid edits that improve only the management fields while the
  text the student sees stays the same.

When student trial results are given:
- check whether the student applied the same judgment to the new video's own observations;
- check whether it copied the example's objects, numbers, directions or conclusion;
- review the soundness of the reasoning and the losses against answering without the example, not only correctness;
- if observations were copied, do not just add exclusions to when_to_use: change the example text and the way it is
  presented;
- two correct examples where the same method gives opposite conclusions may be needed; this is different from
  contrasting a wrong and a fixed solution;
- if it still does not help after revision, you may hold or discard it.

Removing concrete content is not generalization. The criterion is whether the comparison process is reused on new
videos. worked_output must keep the student's format and answer the correct option of its question.

Return JSON: candidates, each with candidate_id, when_to_use, strategy, worked_output, changes, trial_review and
next_action (more_trial: test again unchanged on other questions; revise: test the revised text; memory_review: ready
for the memory manager; hold; discard)."""


# ------------------------------------------------------------------ stage 5
class AddOp(Strict):
    candidate_id: str
    status: Literal['active', 'hold']
    reason: str


class RefineOp(Strict):
    item_id: str
    when_to_use: str
    strategy: str
    worked_output: str
    reason: str


class MergeOp(Strict):
    keep_id: str
    drop_ids: list[str]
    when_to_use: str
    worked_output: str
    reason: str


class PruneOp(Strict):
    item_id: str
    reason: str


class HoldOp(Strict):
    item_id: str
    reason: str


class Stage5Output(Strict):
    block_analysis: str
    add: list[AddOp]
    refine: list[RefineOp]
    merge: list[MergeOp]
    prune: list[PruneOp]
    hold: list[HoldOp]


STAGE5_PROMPT = """You manage the in-context example memory of a small frozen video-language model (Qwen3.5-9B) for one
video benchmark so that it helps on NEW videos. For each new question at most one active item is shown, chosen by
the similarity of its when_to_use to the question; held items are kept but not shown. At most {max_active} items may
be active.

- add: a candidate that covers a useful judgment the memory lacks (status active), or one worth keeping without
  showing yet (status hold). A held item can be re-activated by adding its item_id.
- refine: useful content that the student finds hard to understand or apply; success and corrected items can both be
  rewritten (leave a field empty to keep it).
- merge: items that explain the same judgment twice. Keep conditions that really must be told apart and useful
  contrasting cases.
- prune: items with wrong evidence, items that repeatedly cause wrong imitation, or needless duplicates.
- hold: items whose evidence is too thin to judge now.
Operations are applied in this order: add, prune, hold, merge, refine; a candidate added in the same answer can be
named by its candidate_id in the later operations.

Held items are never shown, so they are never tested on new questions. Activate a candidate whose evidence is
supported and whose trials show no clear harm, even if its benefit is not yet proven: online use on new videos is
how it gets evaluated. Hold or discard when the evidence is unsupported or trials show repeated wrong imitation.

Do not merge items only because they are of the same type: cases with the opposite conclusion or an important
boundary condition are worth keeping separately.

Usage counts and gains/harms are associations influenced by the other inputs and by question difficulty; do not read
them as causal effects of single items. Do not prune after a single loss, and do not treat a single gain as proof.

Give a reason for every change; a rewritten worked_output must answer the correct option of its question.
Return JSON: block_analysis, add, refine, merge, prune, hold."""


# ------------------------------------------------------------------ rendering
def options_text(choices):
    return '\n'.join(f'{k}. {v}' for k, v in choices.items())


def teacher_view(sample):
    """Everything stage 2 may see about a question: frames, question and options. Nothing else exists here."""
    return {'frame_paths': list(sample.frame_paths), 'question': sample.question, 'choices': dict(sample.choices)}


def render_item(item, full=True):
    s = item['stats'].get(str(item['version']), {'shown': 0, 'correct': 0, 'gains': 0, 'harms': 0})
    total = item.get('stats_all', {'shown': 0})
    text = (f'=== item {item["id"]} ({item["kind"]}, {item["status"]}, v{item["version"]}) ===\n'
            f'when_to_use: {item["when_to_use"]}\nstrategy: {item["strategy"]}\n')
    if full:
        text += (f'question:\n{item["question"]}\noptions:\n{options_text(item["choices"])}\n'
                 f'correct option: {item["ground_truth"]}\nworked_output:\n{item["worked_output"]}\n')
    return text + (f'online usage (this version): shown {s["shown"]}, right {s["correct"]}, gains {s["gains"]}, '
                   f'harms {s["harms"]}; all versions: shown {total.get("shown", 0)}')
