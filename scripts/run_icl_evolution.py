"""Self-evolving in-context samples (ICS), one fresh memory per benchmark, full benchmark runs.

Per benchmark the questions come in the vanilla run's order, in blocks of 50:
  block 1   the student answers without samples (= vanilla); the teacher solves every question blind.
  after k   the critic sees block k's cases (student output, correct option, teacher solution, usage) and edits
            the memory: add success / corrected samples, refine, merge, prune. Ops are validated locally.
  block k+1 each question gets up to 5 success + 5 corrected samples whose similarity >= threshold (other videos
            only), as solved examples before the video; with none it is the vanilla input (vanilla output reused).
The three benchmarks run side by side (one GPU call per round); the teacher solves ahead in a thread pool.
Everything is resumable: per-block rows and memory snapshots, cached predictions and teacher responses.
Results are exported after every round to results_dir (each JSONL starts with an accuracy summary line).

    source scripts/icl_env.sh && fuzzy35/bin/python scripts/run_icl_evolution.py [--config ...] [--max-rounds N]
    fuzzy35/bin/python scripts/run_icl_evolution.py --export-only
    fuzzy35/bin/python scripts/run_icl_evolution.py --stress-test 2,4,6   # longest prompts with N samples
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from src.datasets.base_dataset import VideoQADataset  # noqa: E402
from src.evaluation.answer_parser import parse_answer  # noqa: E402
from src.evaluation.correctness import is_correct  # noqa: E402
from src.icl.memory import KINDS, Memory  # noqa: E402
from src.icl.prompts import clean_output, example_block, student_parts, user_prompt  # noqa: E402
from src.icl.retrieval import EmbeddingCache, Retriever, TextEncoder, normalize, query_key  # noqa: E402
from src.icl.stats import paired_stats  # noqa: E402
from src.icl.teacher import (CRITIC_PROMPT, SOLVE_PROMPT, CreditsExhausted, CriticOutput, SolveBatch,  # noqa: E402
                             Teacher, TeacherBudgetExceeded, frame_payload)
from src.utils.storage import append_jsonl, atomic_json, digest, now, read_jsonl  # noqa: E402

FATAL = (CreditsExhausted, TeacherBudgetExceeded)


class Ctx:
    pass


def log(c, msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(c.root / 'run.log', 'a') as f:
        f.write(line + '\n')


# ------------------------------------------------------------------ setup
def setup(config_path, gpu=True, teacher=True, gpus=None):
    c = Ctx()
    c.cfg = cfg = yaml.safe_load(Path(config_path).read_text())
    c.root = Path(cfg['run_root'])
    c.out = Path(cfg['results_dir'])
    c.root.mkdir(parents=True, exist_ok=True)
    c.out.mkdir(parents=True, exist_ok=True)
    frozen = c.root / 'config.json'
    if frozen.exists() and json.loads(frozen.read_text()) != cfg:
        raise RuntimeError(f'{config_path} differs from the config this run started with ({frozen}); use a new run_root')
    atomic_json(frozen, cfg)
    c.system = cfg['student']['system_prompt']
    samples = VideoQADataset(cfg['stream'], 'test').samples
    c.samples = {x.sample_id: x for x in samples}
    c.video = {k: os.path.dirname(x.frame_paths[0]) for k, x in c.samples.items()}
    c.order = {}
    for short, name in cfg['benchmarks'].items():
        c.order[short] = [x.sample_id for x in samples if x.dataset_name == name]
    bs = cfg['block_size']
    c.blocks = {b: [ids[i:i + bs] for i in range(0, len(ids), bs)] for b, ids in c.order.items()}
    c.vanilla = {}  # without a vanilla run (new benchmark), vanilla answers are generated block by block
    if cfg.get('vanilla'):
        c.vanilla = {r[0]: {'raw': r[1], 'answer': r[2], 'valid': bool(r[3]), 'correct': bool(r[4]), 'user': r[5],
                            'system': r[6], 'identity': json.loads(r[7])['backbone_identity']}
                     for r in sqlite3.connect(cfg['vanilla']).execute(
                         'SELECT sample_id, raw_output, parsed_answer, valid, correct, user_prompt, system_prompt, metadata '
                         'FROM experiences')}
        if {v['system'] for v in c.vanilla.values()} != {c.system}:
            raise RuntimeError('vanilla system prompt differs from the configured one')
    c.never_used = {}
    for line in list(open(cfg['baseline_rows']))[1:] if cfg.get('baseline_rows') else []:
        r = json.loads(line)
        c.never_used[r['sample_id']] = r['never_used_question']
    sig = json.loads(Path(cfg['query_encoder_signature']).read_text())
    qcache = EmbeddingCache(cfg['query_cache'], readonly=True)
    media, c.vec = {}, {}
    for k, x in c.samples.items():
        v = qcache.get(query_key(sig, x, cfg['embedding']['query_instruction'], media))
        if v is None:
            raise RuntimeError(f'query vector missing for {k}')
        c.vec[k] = normalize(v)
    text = TextEncoder(cfg['embedding'], EmbeddingCache(c.root / 'text_embeddings.sqlite')) \
        if cfg['retrieval']['key'] == 'when_to_use' else None
    c.retriever = Retriever(c.vec, cfg['retrieval'], text)
    c.db = sqlite3.connect(str(c.root / 'predictions.sqlite'), isolation_level=None, timeout=60)
    c.db.execute('PRAGMA journal_mode=WAL')
    c.db.execute('CREATE TABLE IF NOT EXISTS p (signature TEXT PRIMARY KEY, sample_id TEXT, raw TEXT, answer TEXT, '
                 'valid INTEGER, correct INTEGER, input_tokens INTEGER, generated_tokens INTEGER, seconds REAL, '
                 'created_at TEXT)')
    c.teacher = Teacher(cfg['teacher']['model'], c.root / 'teacher', cfg['teacher'].get('max_calls') or float('inf'),
                        max_usd=cfg['teacher'].get('max_usd'), prices=cfg['teacher'].get('prices')) if teacher else None
    c.pool = None
    if gpu:
        from src.icl.pool import BackbonePool
        s = cfg['student']
        c.pool = BackbonePool(gpus or s['gpus'], s['backbone'], s['video'], s['generation'])
        ref = next(iter(c.vanilla.values()))['identity'] if c.vanilla else c.pool.runtime_identity
        diff = sorted(k for k in set(ref) | set(c.pool.runtime_identity) if k != 'device'
                      and ref.get(k) != c.pool.runtime_identity.get(k))
        if diff:
            raise RuntimeError(f'student runtime differs from the vanilla run in {diff}; vanilla outputs cannot be reused')
    c.solutions = {b: {r['sample_id']: r for r in read_jsonl(c.root / 'teacher_solutions' / f'{b}.jsonl')}
                   if (c.root / 'teacher_solutions' / f'{b}.jsonl').exists() else {} for b in c.order}
    return c


# ------------------------------------------------------------------ state on disk
def block_file(c, kind, b, k):
    return c.root / kind / b / f'block_{k:04d}.json'


def resume_state(c):
    """Next block per benchmark and its memory (snapshot of the last completed block)."""
    nxt, mem = {}, {}
    for b, blocks in c.blocks.items():
        k = 0
        while k < len(blocks) and block_file(c, 'rows', b, k + 1).exists():
            k += 1
        nxt[b] = k + 1
        mem[b] = Memory(b, json.loads(block_file(c, 'memory', b, k).read_text()) if k else [])
    return nxt, mem


# ------------------------------------------------------------------ student
def seed_of(c, sample_id):
    return int(digest({'sample': sample_id, 'seed': c.cfg['seed']})[:8], 16)


def generate(c, plans):
    """plans: [{'sample', 'parts'}] -> results; parts None reuses the vanilla output (identical input)."""
    out, todo = [None] * len(plans), {}
    s = c.cfg['student']
    for i, p in enumerate(plans):
        x = p['sample']
        if p['parts'] is None:
            v = c.vanilla[x.sample_id]
            if v['user'] != user_prompt(x):
                raise RuntimeError(f'vanilla user prompt differs for {x.sample_id}')
            out[i] = {'raw': v['raw'], 'answer': v['answer'], 'valid': v['valid'], 'correct': v['correct'], 'source': 'vanilla'}
            continue
        sig = digest({'sample': x.sample_id, 'parts': p['parts'], 'system': c.system, 'seed': seed_of(c, x.sample_id),
                      'identity': c.pool.runtime_identity, 'generation': s['generation'], 'video': s['video']})
        row = c.db.execute('SELECT raw, answer, valid, correct FROM p WHERE signature=?', (sig,)).fetchone()
        if row:
            out[i] = {'raw': row[0], 'answer': row[1], 'valid': bool(row[2]), 'correct': bool(row[3]), 'source': 'store'}
        else:
            todo.setdefault(sig, (x, p['parts'], []))[2].append(i)
    jobs = list(todo.items())
    for start in range(0, len(jobs), 64):
        chunk = jobs[start:start + 64]
        tasks = [(x.model_input(), '', seed_of(c, x.sample_id), c.system) for _, (x, _, _) in chunk]
        t0 = time.time()
        gens = c.pool.generate_many(tasks, kwargs=[{'parts': parts} for _, (_, parts, _) in chunk])
        per = (time.time() - t0) / max(len(chunk), 1)
        for (sig, (x, parts, idx)), g in zip(chunk, gens):
            parsed = parse_answer(g.raw_output, x.choices)
            ok = bool(is_correct(parsed, x.ground_truth, x.choices))
            c.db.execute('INSERT OR IGNORE INTO p VALUES (?,?,?,?,?,?,?,?,?,?)',
                         (sig, x.sample_id, g.raw_output, parsed.answer, int(parsed.valid), int(ok),
                          g.runtime.get('input_tokens'), g.runtime.get('generated_tokens'), round(per, 2), now()))
            for i in idx:
                out[i] = {'raw': g.raw_output, 'answer': parsed.answer, 'valid': parsed.valid, 'correct': ok,
                          'source': 'generated'}
    return out


# ------------------------------------------------------------------ teacher
def solve_chunk(c, b, ids):
    """Blind teacher solutions for up to solve_batch questions (same 32 frames as the student)."""
    from src.datasets.video_loader import load_frames
    eids = {k: f'E{i + 1}' for i, k in enumerate(ids)}
    blocks = [('text', 'Experiences follow, each with an experience_id, the question, options and 32 frames.')]
    for k in ids:
        x = c.samples[k]
        frames, meta = load_frames(x.frame_paths, **c.cfg['student']['video'])
        blocks.append(('text', f'\n=== experience_id {eids[k]} ===\nQUESTION: {x.question}\nOPTIONS:\n'
                       + '\n'.join(f'{a}. {t}' for a, t in x.choices.items()) + '\nFRAMES:'))
        blocks += [('image', label, url) for label, url in frame_payload(frames, meta['sampled_fps'])]
    try:
        got = {s.experience_id: s for s in c.teacher.call('solve', SOLVE_PROMPT, blocks, SolveBatch).solutions}
        error = None
    except FATAL:
        raise
    except Exception as exc:  # recorded as missing; the case then cannot become a sample
        got, error = {}, f'{type(exc).__name__}: {exc}'[:300]
    rows = []
    for k in ids:
        s = got.get(eids[k])
        x = c.samples[k]
        sol = s.model_dump() if s else None
        answer = sol['answer'].strip().upper() if sol else None
        rows.append({'sample_id': k, 'solution': sol, 'answer': answer, 'correct': answer == x.ground_truth if sol else None,
                     'error': error if not sol else None})
    path = c.root / 'teacher_solutions' / f'{b}.jsonl'
    for r in rows:
        append_jsonl(path, r)
        c.solutions[b][r['sample_id']] = r
    return rows


class Prefetch:
    """Teacher solves run ahead of the student, earliest block first."""

    def __init__(self, c):
        self.c, self.pool, self.futures = c, ThreadPoolExecutor(c.cfg['teacher']['threads']), {}

    def ensure(self, nxt):
        c = self.c
        n = c.cfg['teacher']['solve_batch']
        for ahead in range(c.cfg['teacher']['lookahead_rounds']):
            for b, k in nxt.items():
                if k + ahead > len(c.blocks[b]):
                    continue
                ids = c.blocks[b][k + ahead - 1]
                for j in range(0, len(ids), n):
                    chunk = ids[j:j + n]
                    key = (b, k + ahead, j)
                    if key in self.futures or all(i in c.solutions[b] for i in chunk):
                        continue
                    self.futures[key] = self.pool.submit(solve_chunk, c, b, chunk)

    def wait(self, b, k):
        for key in sorted(x for x in self.futures if x[:2] == (b, k)):
            self.futures.pop(key).result()  # FATAL errors propagate


# ------------------------------------------------------------------ critic
def clip(text, n):
    text = (text or '').strip()
    return text if len(text) <= n else text[:n] + ' [...]'


def render_item(i):
    s = i['stats']
    return (f'=== item {i["id"]} ({i["kind"]}, v{i["version"]}) ===\nwhen_to_use: {i["when_to_use"]}\nstrategy: {i["strategy"]}\n'
            f'question: {clip(i["question"], 400)}\ncorrect option text: {i["choices"][i["ground_truth"]]}\n'
            f'example: {clip(i["example"], 700)}\nusage: shown {s["shown"]}, model right {s["correct_when_shown"]}, '
            f'gains {s["gains"]}, harms {s["harms"]}')


def memory_view(c, mem, cases):
    active = mem.active()
    shown = {i for case in cases.values() for i in case['shown_ids']}
    pick = {i['id'] for i in active if i['id'] in shown or i['stats']['harms'] > i['stats']['gains']}
    for case in cases.values():
        if case['teacher_correct']:
            q = c.vec[case['sample_id']]
            near = sorted(active, key=lambda i: -float(c.vec[i['sample_id']] @ q))[:3]
            pick |= {i['id'] for i in near}
    ids = sorted(pick)[:c.cfg['critic']['memory_view_max']]
    return [mem.items[i] for i in ids], len(active)


def render_case(c, cid, case):
    t = case['teacher_solution']
    teacher = 'no solution (call failed)' if not t else (
        f'answer {t["answer"]} ({"right" if case["teacher_correct"] else "wrong"}), confidence {t["confidence"]}; facts: '
        + '; '.join(f'[frames {f["frames"]}] {f["fact"]} ({f["certainty"]})' for f in t['facts'][:5])
        + f'; relation: {t["relation"]}')
    model = (f'answer {case["answer"] or "invalid"} ({"right" if case["correct"] else "wrong"}); without examples: '
             f'{"right" if case["vanilla_correct"] else "wrong"}; shown items: {", ".join(case["shown_ids"]) or "none"}')
    return (f'=== case {cid} ===\nquestion: {clip(case["question"], 900)}\noptions:\n'
            + '\n'.join(f'{a}. {v}' for a, v in case['choices'].items())
            + f'\ncorrect option: {case["ground_truth"]}\nmodel: {model}\nmodel output: '
            f'{clip(case["output"], c.cfg["critic"]["case_output_chars"])}\nyour blind solution: {teacher}')


def critic(c, b, k, mem, cases):
    view, n_active = memory_view(c, mem, cases)
    counts = Counter(i['kind'] for i in mem.active())
    first, last = (k - 1) * c.cfg['block_size'] + 1, (k - 1) * c.cfg['block_size'] + len(cases)
    text = (f'BENCHMARK: {c.cfg["benchmarks"][b]}   BLOCK {k} (questions {first}-{last})\n'
            f'MEMORY: {n_active} active items ({counts["success"]} success, {counts["corrected"]} corrected); '
            f'{len(view)} relevant items shown below.\n\n' + '\n\n'.join(render_item(i) for i in view)
            + '\n\nCASES:\n\n' + '\n\n'.join(render_case(c, cid, case) for cid, case in cases.items()))
    prompt = CRITIC_PROMPT.replace('{max_add}', str(c.cfg['critic']['max_add_per_kind']))
    return c.teacher.call('critic', prompt, [('text', text)], CriticOutput), [i['id'] for i in view]


# ------------------------------------------------------------------ one round
def plan_block(c, b, k, mem):
    items = mem.active()
    plans = []
    for sid in c.blocks[b][k - 1]:
        x = c.samples[sid]
        selected, ranking = c.retriever.retrieve(x, c.video[sid], items)
        # GPU memory: drop the least similar samples until the example text fits (10 long Urban samples OOM on 11 GB)
        while selected and len(example_block(sid, [i for i, _ in selected], False)) > c.cfg['retrieval']['max_example_chars']:
            selected.remove(min(selected, key=lambda t: t[1]))
        chosen = [i for i, _ in selected]
        plans.append({'sample': x, 'parts': student_parts(x, chosen, c.cfg['retrieval']['permute_options']),
                      'shown': [{'id': i['id'], 'kind': i['kind'], 'version': i['version'], 'score': round(s, 4)}
                                for i, s in selected], 'ranking': ranking})
    return plans


def finish_block(c, b, k, mem, plans, results):
    rows, cases = [], {}
    counts = Counter(i['kind'] for i in mem.active())
    for j, (p, r) in enumerate(zip(plans, results), 1):
        x = p['sample']
        sol = c.solutions[b].get(x.sample_id)
        v = c.vanilla[x.sample_id]
        row = {'benchmark': b, 'block': k, 'index': (k - 1) * c.cfg['block_size'] + j, 'sample_id': x.sample_id,
               'video': c.video[x.sample_id], 'never_used_question': c.never_used.get(x.sample_id),
               'memory_active': dict(counts), 'shown': p['shown'], 'ranking': p['ranking'],
               'shown_ids': [s['id'] for s in p['shown']],
               'student': {'answer': r['answer'], 'correct': r['correct'], 'valid': r['valid'], 'output': r['raw'],
                           'source': r['source']},
               'vanilla': {'answer': v['answer'], 'correct': v['correct'], 'valid': v['valid']},
               'teacher': {'answer': sol['answer'] if sol else None, 'correct': sol['correct'] if sol else None,
                           'confidence': sol['solution']['confidence'] if sol and sol['solution'] else None},
               'correct': r['correct'], 'vanilla_correct': v['correct']}
        rows.append(row)
        cases[f'Q{j:02d}'] = {'sample_id': x.sample_id, 'video': c.video[x.sample_id], 'question': x.question,
                              'choices': x.choices, 'ground_truth': x.ground_truth, 'output': r['raw'],
                              'answer': r['answer'], 'valid': r['valid'], 'correct': r['correct'],
                              'vanilla_correct': v['correct'], 'shown_ids': row['shown_ids'],
                              'teacher_solution': sol['solution'] if sol else None,
                              'teacher_answer': sol['answer'] if sol else None,
                              'teacher_correct': bool(sol and sol['correct']),
                              'teacher_confidence': sol['solution']['confidence'] if sol and sol['solution'] else None}
    mem.record_usage(rows)
    record = {'benchmark': b, 'block': k, 'skipped': k == len(c.blocks[b]), 'applied': [], 'rejected': []}
    if not record['skipped']:  # the last block's samples could never be used
        try:
            out, view = critic(c, b, k, mem, cases)
            applied, rejected = mem.apply(out, cases, k, c.cfg['critic']['max_add_per_kind'])
            record.update(analysis=out.block_analysis, applied=applied, rejected=rejected, memory_view=view)
        except FATAL:
            raise
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'[:500]
    record['memory_after'] = dict(Counter(i['kind'] for i in mem.active()))
    atomic_json(block_file(c, 'critic', b, k), record)
    atomic_json(block_file(c, 'memory', b, k), mem.snapshot())
    atomic_json(block_file(c, 'rows', b, k), rows)  # written last: marks the block complete
    return rows, record


def run(c, max_rounds=None):
    nxt, mem = resume_state(c)
    log(c, f'start: next blocks {nxt}; active samples ' + str({b: len(m.active()) for b, m in mem.items()}))
    pre = Prefetch(c)
    rounds = 0
    while True:
        active = {b: k for b, k in nxt.items() if k <= len(c.blocks[b])}
        if not active or (max_rounds is not None and rounds >= max_rounds):
            break
        pre.ensure(active)
        t0 = time.time()
        plans = {b: plan_block(c, b, k, mem[b]) for b, k in active.items()}
        flat = [p for b in active for p in plans[b]]
        results = generate(c, flat)
        t_gen = time.time() - t0
        by, i = {}, 0
        for b in active:
            by[b] = results[i:i + len(plans[b])]
            i += len(plans[b])
        for b, k in active.items():
            pre.wait(b, k)
        with ThreadPoolExecutor(len(active)) as ex:
            done = {b: ex.submit(finish_block, c, b, k, mem[b], plans[b], by[b]) for b, k in active.items()}
            done = {b: f.result() for b, f in done.items()}
        for b, (rows, record) in done.items():
            n = len(rows)
            log(c, f'{b} block {active[b]}/{len(c.blocks[b])}: icl {sum(r["correct"] for r in rows)}/{n} '
                f'vanilla {sum(r["vanilla_correct"] for r in rows)}/{n} with samples {sum(bool(r["shown"]) for r in rows)} '
                f'| ops {Counter(a["op"] for a in record["applied"])} rejected {len(record["rejected"])}'
                f'{" ERROR " + record["error"] if record.get("error") else ""} | memory {record["memory_after"]}')
            nxt[b] += 1
        log(c, f'round {rounds + 1}: generation {t_gen / 60:.1f} min, total {(time.time() - t0) / 60:.1f} min; '
            f'teacher calls {c.teacher.used}')
        export(c)
        rounds += 1
    pre.pool.shutdown(wait=False, cancel_futures=True)
    return nxt


# ------------------------------------------------------------------ export
def summarize(rows, samples, items_by_id):
    def a(vals):
        vals = [v for v in vals if v is not None]
        return {'accuracy': round(sum(vals) / len(vals), 4) if vals else None, 'correct': int(sum(vals)), 'n': len(vals)}

    def paired(sub):
        st = paired_stats([r['vanilla_correct'] for r in sub], [r['correct'] for r in sub], [r['video'] for r in sub])
        return {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v] if isinstance(v, list) else v)
                for k, v in st.items()}
    with_s = [r for r in rows if r['shown']]
    after1 = [r for r in rows if r['block'] > 1]
    fresh = [r for r in rows if r['never_used_question']]

    def text_match(r):
        gt = samples[r['sample_id']]
        gt_text = gt.choices[gt.ground_truth].strip().lower()
        return any(items_by_id[s['id']]['choices'][items_by_id[s['id']]['ground_truth']].strip().lower() == gt_text
                   for s in r['shown'] if s['id'] in items_by_id)
    match = [r for r in with_s if text_match(r)]
    return {'questions': len(rows), 'blocks': len({r['block'] for r in rows}),
            'accuracy': {'icl': {**a([r['correct'] for r in rows]), 'invalid': sum(not r['student']['valid'] for r in rows)},
                         'vanilla': {**a([r['vanilla_correct'] for r in rows]), 'invalid': sum(not r['vanilla']['valid'] for r in rows)},
                         'teacher': a([r['teacher']['correct'] for r in rows])},
            'icl_vs_vanilla': paired(rows),
            'icl_vs_vanilla_from_block2': paired(after1),
            'questions_with_samples': {'n': len(with_s), 'mean_shown': round(np.mean([len(r['shown']) for r in with_s]), 2) if with_s else 0,
                                       'icl_vs_vanilla': paired(with_s)},
            'never_used_questions': paired(fresh),
            'shown_sample_answer_text_equals_ground_truth_text': {'n': len(match), 'icl_vs_vanilla': paired(match)},
            'rule': 'accuracy = correct / questions; invalid output counts as wrong; icl = first answer with the memory '
                    'frozen before the block; vanilla = same student, same question, no samples'}


def write_jsonl(path, header, rows):
    tmp = path.with_suffix('.jsonl.tmp')
    with open(tmp, 'w') as f:
        f.write(json.dumps(header, ensure_ascii=False) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def export(c):
    overall, all_rows = [], []
    for b, blocks in c.blocks.items():
        rows = []
        for k in range(1, len(blocks) + 1):
            p = block_file(c, 'rows', b, k)
            if not p.exists():
                break
            rows += json.loads(p.read_text())
        if not rows:
            continue
        last = rows[-1]['block']
        items = json.loads(block_file(c, 'memory', b, last).read_text())
        by_id = {i['id']: i for i in items}
        for k in range(1, last + 1):  # samples shown earlier may have been rewritten since: use their own versions
            for i in json.loads(block_file(c, 'memory', b, k).read_text()):
                by_id.setdefault(i['id'], i)
        s = summarize(rows, c.samples, by_id)
        out_rows = []
        for r in rows:
            x = c.samples[r['sample_id']]
            out_rows.append({'index': r['index'], 'block': r['block'], 'sample_id': r['sample_id'], 'video': r['video'],
                             'question': x.question, 'choices': x.choices, 'ground_truth': x.ground_truth,
                             'shown_samples': r['shown'], 'student': r['student'], 'vanilla': r['vanilla'],
                             'teacher': r['teacher'], 'never_used_question': r['never_used_question'],
                             'memory_active': r['memory_active'], 'top_candidates': r['ranking']})
        write_jsonl(c.out / f'model_output_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b],
                    'total_questions': len(c.order[b]), **s}, out_rows)
        for kind, name in (('success', 'succ'), ('corrected', 'corr')):
            ks = [i for i in items if i['kind'] == kind]
            act = [i for i in ks if i['status'] == 'active']
            shown = sum(i['stats']['shown'] for i in ks)
            write_jsonl(c.out / f'{name}_ics_{b}.jsonl', {
                '_type': 'summary', 'benchmark': c.cfg['benchmarks'][b], 'kind': kind, 'after_block': last,
                'items': len(ks), 'active': len(act), 'status': dict(Counter(i['status'] for i in ks)),
                'times_shown': shown, 'student_accuracy_when_shown': round(sum(i['stats']['correct_when_shown'] for i in ks) / shown, 4) if shown else None,
                'gains': sum(i['stats']['gains'] for i in ks), 'harms': sum(i['stats']['harms'] for i in ks),
                'note': 'gains/harms: questions where the item was shown and the student changed from wrong to right '
                        '(or right to wrong) against vanilla; credit is shared by all items shown (association)'}, ks)
        ops = [json.loads(block_file(c, 'critic', b, k).read_text()) for k in range(1, last + 1)]
        tally = Counter(a['op'] for o in ops for a in o['applied'])
        write_jsonl(c.out / f'critic_ops_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b], 'blocks': last,
                    'applied': dict(tally), 'rejected': sum(len(o['rejected']) for o in ops),
                    'critic_errors': sum(1 for o in ops if o.get('error')), 'accuracy': s['accuracy']}, ops)
        curve = []
        for k in range(1, last + 1):
            br = [r for r in rows if r['block'] == k]
            curve.append({'block': k, 'n': len(br), 'icl': round(np.mean([r['correct'] for r in br]), 4),
                          'vanilla': round(np.mean([r['vanilla_correct'] for r in br]), 4),
                          'teacher': round(np.mean([r['teacher']['correct'] for r in br if r['teacher']['correct'] is not None]), 4)
                          if any(r['teacher']['correct'] is not None for r in br) else None,
                          'with_samples': sum(bool(r['shown']) for r in br),
                          'mean_shown': round(np.mean([len(r['shown']) for r in br]), 2), 'memory_active': br[0]['memory_active']})
        write_jsonl(c.out / f'block_curve_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b],
                    'accuracy': s['accuracy'], 'icl_vs_vanilla': s['icl_vs_vanilla']}, curve)
        overall.append({'benchmark': c.cfg['benchmarks'][b], **s})
        all_rows += rows
    if not all_rows:
        return
    total = summarize(all_rows, c.samples, {})
    total.pop('shown_sample_answer_text_equals_ground_truth_text')
    gens = c.db.execute('SELECT COUNT(*), SUM(input_tokens), SUM(generated_tokens), SUM(seconds) FROM p').fetchone()
    cost = {'teacher_calls_reserved': c.teacher.used if c.teacher else None,
            'teacher_by_kind': c.teacher.usage() if c.teacher else None,
            'student_generations': gens[0], 'student_input_tokens': gens[1], 'student_generated_tokens': gens[2],
            'student_gpu_seconds_per_generation_x4gpus': round(gens[3] / gens[0], 2) if gens[0] else None,
            'reused_vanilla_outputs': sum(r['student']['source'] == 'vanilla' for r in all_rows),
            'note': 'vanilla outputs are needed for the gains/harms the critic sees: one extra student generation per '
                    'question in a deployed system (here taken from the existing vanilla run)'}
    write_jsonl(c.out / 'overall_accuracy.jsonl', {'_type': 'summary', 'benchmark': 'all', 'updated_at': now(),
                'total_questions': sum(len(v) for v in c.order.values()), **total, 'cost': cost,
                'config': {k: c.cfg[k] for k in ('block_size', 'retrieval', 'critic', 'teacher')}}, overall)


# ------------------------------------------------------------------ stress test
def stress_test(c, counts):
    """Longest inputs the run can build: n of the longest Urban questions+outputs as samples, on the 4 longest questions."""
    urban = [c.samples[k] for k in c.order['urban']]
    longest = sorted(urban, key=lambda x: -(len(x.question) + len(c.vanilla[x.sample_id]['raw'])))
    pool = [{'id': f'stress-{i}', 'question': x.question, 'choices': x.choices,
             'example': clean_output(c.vanilla[x.sample_id]['raw']) or f'<THINK>.</THINK>\n<ANSWER>{x.ground_truth}</ANSWER>'}
            for i, x in enumerate(longest[4:14])]
    targets = longest[:4]
    for n in counts:
        items = pool[:n]
        chars = len(student_parts(targets[0], items)[0])
        tasks = [(x.model_input(), '', seed_of(c, x.sample_id), c.system) for x in targets]
        try:
            gens = c.pool.generate_many(tasks, kwargs=[{'parts': student_parts(x, items)} for x in targets])
        except RuntimeError as exc:
            print(f'n={n} example chars {chars}: FAILED ({"OutOfMemory" if "out of memory" in str(exc) else str(exc)[-200:]})', flush=True)
            continue
        print(f'n={n} example chars {chars}: input tokens {max(g.runtime["input_tokens"] for g in gens)}, peak GiB '
              f'{max(g.runtime["gpu_peak_allocated_bytes"] for g in gens) / 2 ** 30:.2f}, valid '
              f'{sum(parse_answer(g.raw_output, x.choices).valid for x, g in zip(targets, gens))}/4', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/icl_evolution.yaml')
    ap.add_argument('--max-rounds', type=int)
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--stress-test', type=lambda v: [int(n) for n in v.split(',')], metavar='N,N,...')
    a = ap.parse_args()
    if a.export_only:
        export(setup(a.config, gpu=False, teacher=True))
        return
    c = setup(a.config, gpu=True, teacher=not a.stress_test)
    try:
        if a.stress_test:
            stress_test(c, a.stress_test)
        else:
            nxt = run(c, a.max_rounds)
            log(c, f'stopped: next blocks {nxt}')
    finally:
        c.pool.close()


if __name__ == '__main__':
    main()
