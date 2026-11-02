"""Teacher and critic calls (gpt-6.1-sol, OpenAI Responses, strict JSON schema).

The teacher solves every question blind from the same 32 frames the student saw (no answer, no student output).
The critic is text only: after a block's answers are scored it sees the block's cases and the memory, and returns
memory operations. Every API attempt counts toward the call cap; responses are cached by request digest, so a
resumed run replays identical requests for free.
"""
import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict
from src.utils.storage import append_jsonl, atomic_json, digest, now, read_jsonl


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Fact(Strict):
    fact: str
    frames: str
    certainty: Literal['clear', 'likely', 'uncertain']


class Solution(Strict):
    experience_id: str
    answer: str
    evidence_frames: list[int]
    facts: list[Fact]
    relation: str
    confidence: Literal['high', 'medium', 'low']


class SolveBatch(Strict):
    solutions: list[Solution]


class AddOp(Strict):
    kind: Literal['success', 'corrected']
    case_id: str
    when_to_use: str
    strategy: str
    corrected_output: str
    reason: str


class RefineOp(Strict):
    item_id: str
    when_to_use: str
    strategy: str
    corrected_output: str
    reason: str


class MergeOp(Strict):
    keep_id: str
    drop_ids: list[str]
    when_to_use: str
    reason: str


class PruneOp(Strict):
    item_id: str
    reason: str


class CriticOutput(Strict):
    block_analysis: str
    add: list[AddOp]
    refine: list[RefineOp]
    merge: list[MergeOp]
    prune: list[PruneOp]


SOLVE_PROMPT = """Solve multiple-choice questions about videos. For each experience you see the question, the options and
32 frames in temporal order (frame number and time). Answer from what you can verify in the frames:
- answer: one option letter;
- evidence_frames: the frame numbers (1-32) that decide the answer;
- facts: the observations you used, each with its frame numbers; only what is visible;
- relation: one or two sentences on how these facts select the answer over the other options;
- confidence: high / medium / low.
Return JSON with one solution per experience_id."""

CRITIC_PROMPT = """You maintain the in-context example memory of a small frozen video-language model (Qwen3.5-9B, 32 frames
of about 256x256 pixels) for one video benchmark. Before the model answers a new question, the memory items most
similar to that question (same benchmark, other videos) are shown to it as solved examples: the past question, its
options and a worked answer in the model's own format (<THINK>...</THINK> then <ANSWER>letter</ANSWER>). The example
videos are not shown, and the example options are relabelled at random, so only the reasoning can transfer.

Item kinds:
- success: a question the model answered correctly while you, solving blind, also answered correctly. The example
  is the model's own output, unchanged.
- corrected: a question the model got wrong while you answered correctly. The example is the model's own output
  minimally corrected: keep its wording, style and length (within about 30%); change only the wrong observation
  and the link from what is seen to the correct option; state only what your blind solution saw in the frames.

You receive the current memory (relevant items) with usage counts: times shown, how often the model was right when
shown, gains and harms against the same model answering the same question without examples (associations, not
causes). You also receive this block's cases: question, options, correct option, the model's output, its
correctness, its correctness without examples, the items it was shown, and your blind solution with confidence.

Update the memory so that it helps on NEW videos:
- add: items from this block's cases that show a reusable way of looking at the video or of linking what is seen
  to the options, not facts specific to that video. At most {max_add} per kind. Prefer error patterns that recur and
  question types the memory does not cover yet. success needs a case where both the model and you were right;
  corrected needs a case where the model was wrong (with a valid answer) and you were right. Skip cases where the
  needed information is not visible or your solution is uncertain. For corrected, write corrected_output; for
  success leave corrected_output empty (the model's own output is used).
- refine: rewrite an item's when_to_use, strategy or (corrected only) example when usage shows that it is retrieved
  for the wrong questions or misleads (for example its scene details or its conclusion get copied).
- merge: items that teach the same thing for the same kind of question: keep the clearest, give it a when_to_use
  covering both, drop the others (same kind only).
- prune: items with clearly more harms than gains, redundant items, or misleading ones.
when_to_use: one or two sentences on the kind of question and visual situation where the item helps (question type,
what must be observed, the typical trap). strategy: one sentence naming the reasoning step the example shows. Both
are for maintaining the memory; neither is shown to the model. Leave a refine field empty to keep it.
Change nothing without a clear reason. Return JSON."""


def frame_payload(frames, fps):
    """The 32 RGB frames the student received, as JPEG data URLs with times (same pixels, same order)."""
    from PIL import Image
    out = []
    for i, frame in enumerate(frames):
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=90)
        out.append((f'frame {i + 1}, t={i / fps:.1f}s', 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()))
    return out


class TeacherBudgetExceeded(RuntimeError):
    pass


class CreditsExhausted(RuntimeError):
    pass


class Teacher:
    def __init__(self, model, root, max_calls, timeout=900, attempts=4, wait_seconds=30, max_usd=None, prices=None):
        """max_usd: stop (TeacherBudgetExceeded) once the logged spend reaches it; prices: USD per 1M tokens
        {input, cached_input, output}. Spend counts every logged response of this root, so it survives a restart."""
        from openai import OpenAI
        from scripts.api_credentials import load_key_into_environment
        load_key_into_environment()
        self.client = OpenAI(timeout=timeout, max_retries=0)
        self.model, self.root, self.max_calls = model, Path(root), max_calls
        self.attempts, self.wait_seconds = attempts, wait_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.root / 'calls.jsonl'
        self.lock = threading.Lock()
        self.used = len(read_jsonl(self.calls_path)) if self.calls_path.exists() else 0
        self.max_usd, self.prices = max_usd, prices
        self.local = threading.local()  # cost of this thread's last call (0 for a cache hit)
        results = self.root / 'results.jsonl'
        self.spent = sum(self.cost(r.get('usage') or {}) for r in read_jsonl(results)) if prices and results.exists() else 0.0

    def cost(self, usage):
        if not self.prices:
            return 0.0
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
        return ((usage.get('input_tokens', 0) - cached) * self.prices['input'] + cached * self.prices['cached_input']
                + usage.get('output_tokens', 0) * self.prices['output']) / 1e6

    def call(self, kind, system, blocks, schema, max_output_tokens=16000):
        """blocks: list of ('text', str) | ('image', label, data_url)."""
        key = digest({'model': self.model, 'system': system, 'schema': schema.model_json_schema(),
                      'blocks': [b if b[0] == 'text' else ('image', b[1], digest(b[2])) for b in blocks]})
        cache = self.root / 'cache' / f'{key}.json'
        self.local.cost = 0.0
        if cache.exists():
            return schema.model_validate(json.loads(cache.read_text())['response'])
        content = []
        for b in blocks:
            if b[0] == 'text':
                content.append({'type': 'input_text', 'text': b[1]})
            else:
                content += [{'type': 'input_text', 'text': b[1]}, {'type': 'input_image', 'image_url': b[2], 'detail': 'low'}]
        for attempt in range(self.attempts):
            with self.lock:  # reserve the call before sending it, so parallel calls cannot pass the cap
                if self.used >= self.max_calls:
                    raise TeacherBudgetExceeded(f'teacher call cap {self.max_calls} reached')
                if self.max_usd is not None and self.spent >= self.max_usd:
                    raise TeacherBudgetExceeded(f'teacher spend cap ${self.max_usd} reached (${self.spent:.2f})')
                self.used += 1
                append_jsonl(self.calls_path, {'kind': kind, 'key': key, 'attempt': attempt, 'timestamp': now()})
            start = time.time()
            record = {'kind': kind, 'key': key, 'attempt': attempt, 'timestamp': now()}
            try:
                response = self.client.responses.create(model=self.model, input=[
                    {'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                    text={'format': {'type': 'json_schema', 'name': kind, 'strict': True,
                                     'schema': schema.model_json_schema()}}, max_output_tokens=max_output_tokens)
                usage = response.usage.model_dump() if response.usage else {}
                with self.lock:
                    self.local.cost = self.cost(usage)
                    self.spent += self.local.cost
                parsed = schema.model_validate_json(response.output_text)
                append_jsonl(self.root / 'results.jsonl', dict(record, ok=True, usage=usage, seconds=round(time.time() - start, 1)))
                atomic_json(cache, {'kind': kind, 'response': parsed.model_dump(), 'usage': usage}, immutable=False)
                return parsed
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'[:500]
                append_jsonl(self.root / 'results.jsonl', dict(record, ok=False, error=message))
                if 'insufficient_quota' in message or 'credit' in message.lower():
                    raise CreditsExhausted(message) from exc
                if attempt == self.attempts - 1:
                    raise
                time.sleep(self.wait_seconds * (attempt + 1))

    def usage(self):
        """Calls, attempts and tokens by kind, from the results log."""
        out = {}
        for r in read_jsonl(self.root / 'results.jsonl') if (self.root / 'results.jsonl').exists() else []:
            k = out.setdefault(r['kind'], {'ok': 0, 'failed': 0, 'input_tokens': 0, 'output_tokens': 0})
            if r.get('ok'):
                k['ok'] += 1
                k['input_tokens'] += r['usage'].get('input_tokens', 0)
                k['output_tokens'] += r['usage'].get('output_tokens', 0)
            else:
                k['failed'] += 1
        return out
