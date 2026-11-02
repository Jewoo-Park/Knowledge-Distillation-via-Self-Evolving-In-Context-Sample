"""Five-stage ICS evolution: the student answers a block with the current memory (first answers are the score), then
  1 select   the critic picks up to N cases worth a visual re-check (text only)
  2 solve    the teacher solves only those cases blind: frames + question + options (teacher_view, nothing else)
  3 compare  the critic compares student and teacher solutions and writes worked-example candidates
  4 refine   write when_to_use / strategy -> try each candidate on past questions of other videos -> review, repeat
  5 manage   the critic adds / refines / merges / prunes / holds memory items
The updated memory applies from the next block. Benchmarks run in parallel threads (GPU calls serialized); each
keeps its own memory. Resumable: a block is done when rows/<b>/block_k.json exists; teacher/critic responses and
student generations are cached by input, so a restart replays without new cost.

    source scripts/icl_env.sh && fuzzy35/bin/python scripts/run_icl_stages.py [--max-blocks N] [--benchmarks urban,...]
    fuzzy35/bin/python scripts/run_icl_stages.py --export-only
"""
import argparse
import json
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_icl_evolution as R  # noqa: E402  (setup, student generation; sets cwd and sys.path)
import numpy as np  # noqa: E402
from src.icl.prompts import VIDEO, example_block, user_prompt  # noqa: E402
from src.icl.stage_memory import StageMemory, answers  # noqa: E402
from src.icl.stages import (STAGE1_PROMPT, STAGE2_PROMPT, STAGE3_PROMPT, STAGE4_PROMPT, STAGE5_PROMPT,  # noqa: E402
                            Stage1Output, Stage2Output, Stage3Output, Stage4Output, Stage5Output, options_text,
                            render_item, teacher_view)
from src.icl.stats import paired_stats  # noqa: E402
from src.icl.teacher import CreditsExhausted, TeacherBudgetExceeded, frame_payload  # noqa: E402
from src.utils.storage import atomic_json, digest, now  # noqa: E402

FATAL = (CreditsExhausted, TeacherBudgetExceeded)
AFTER_FOOTER = 'Now answer the question about the video above.\n\n'


def norm(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def log(c, msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with c.log_lock, open(c.root / 'run.log', 'a') as f:
        f.write(line + '\n')


def setup(config, gpu=True, gpus=None):
    c = R.setup(config, gpu=gpu, teacher=True, gpus=gpus)
    import sqlite3
    c.db = sqlite3.connect(str(c.root / 'predictions.sqlite'), isolation_level=None, timeout=60, check_same_thread=False)
    c.gpu_lock, c.enc_lock, c.log_lock = threading.Lock(), threading.Lock(), threading.Lock()
    c.past_top = {b: [] for b in c.blocks}
    if c.cfg['retrieval'].get('mode') == 'coverage':
        # up to max_examples best matches, shown when the top-1 score is at least the (1 - coverage) quantile of the
        # top-1 scores of the questions answered before (past information only; nothing is shown before min_history)
        c.rule = {b: {'mode': 'coverage', 'coverage': c.cfg['retrieval']['coverage'], 'threshold': None} for b in c.blocks}
        return c
    if c.cfg['retrieval'].get('mode') == 'absolute':  # one fixed top-1 threshold, no margin (the C2 rule)
        c.rule = {b: {'mode': 'absolute', 'threshold': c.cfg['retrieval']['threshold']} for b in c.blocks}
        return c
    cal_path = Path(c.cfg['retrieval']['calibration'])
    cal = json.loads(cal_path.read_text())['benchmarks'] if cal_path.exists() else {}
    c.rule = {}
    for b in c.blocks:
        chosen = cal.get(b, {}).get('chosen', {})
        thr = c.cfg['retrieval']['threshold'] if c.cfg['retrieval']['threshold'] is not None else chosen.get('threshold')
        mar = c.cfg['retrieval']['margin'] if c.cfg['retrieval']['margin'] is not None else chosen.get('margin')
        if thr is None or mar is None:
            raise RuntimeError(f'no retrieval threshold/margin for {b}: run scripts/calibrate_wtu_retrieval.py')
        c.rule[b] = {'threshold': thr, 'margin': mar}
    return c


def wtu_vector(c, text):
    with c.enc_lock:
        return c.retriever.text.encode(text)


def gen(c, plans):
    with c.gpu_lock:
        return R.generate(c, plans)


def call(c, kind, prompt, blocks, schema):
    return c.teacher.call(kind, prompt, blocks, schema)


def block_path(c, kind, b, k, suffix='json'):
    return c.root / kind / b / f'block_{k:04d}.{suffix}'


# ------------------------------------------------------------------ student input
def item_view(item):
    return {'id': item['id'], 'question': item['question'], 'choices': item['choices'], 'example': item['worked_output']}


def parts_for(c, x, item, salt=''):
    """Student content with one sample (or a list of samples); options of each sample relabelled at random (letters
    inside follow). One sample gives the same text whether passed alone or as a one-item list."""
    views = [dict(item_view(i), id=i['id'] + salt) for i in (item if isinstance(item, list) else [item])]
    text = example_block(x.sample_id, views, c.cfg['retrieval']['permute_options'])
    if c.cfg['retrieval']['position'] == 'after_video':
        return [VIDEO, text.replace('Now answer the new question about the following video.\n\n', AFTER_FOOTER) + user_prompt(x)]
    return [text, VIDEO, user_prompt(x)]


def video_group(path):
    """Original video of a clip directory: AerialVLN episodes of one trajectory share the hash; clips drop the _N suffix
    (the rule of scripts/ics_policy_search.video_group)."""
    stem = Path(path).name
    m = re.match(r'AerialVLN_\d+_([A-Z0-9]+)(?:_\d+)?$', stem)
    if m:
        return 'AerialVLN_' + m.group(1)
    m = re.match(r'(EmbodiedCity_\d+|RealWorld_GN_\d+|RealWorld_RN_\d+|RealWorld_\d+)(?:_\d+)?$', stem)
    return m.group(1) if m else stem


def eligible(c, x, items):
    """Drops items of the same clip (exclude: clip, run1) or of the same original video (exclude: group), the identical
    question, and (exclude_option_text) items whose correct option text is one of the current options."""
    options = {norm(t) for t in x.choices.values()}
    same = (x.question.strip(), tuple(sorted(x.choices.items())))
    group = c.cfg['retrieval'].get('exclude', 'clip') == 'group'
    out = []
    for i in items:
        if i['video'] == c.video[x.sample_id] or (i['question'].strip(), tuple(sorted(i['choices'].items()))) == same:
            continue
        if group and video_group(i['video']) == video_group(c.video[x.sample_id]):
            continue
        if c.cfg['retrieval']['exclude_option_text'] and norm(i['choices'][i['ground_truth']]) in options:
            continue
        out.append(i)
    return out


def retrieve(c, b, x, items):
    """-> (shown items, ranking). Ranked by when_to_use cosine, or (key: question) by the cosine of this
    question's query vector with that of the question each item was written from. calibrated: the top-1 above the
    threshold and the margin; coverage: the max_examples best when the top-1 is above the current coverage threshold;
    absolute: the top-1 when its score is at least the fixed threshold."""
    pool = eligible(c, x, items)
    if not pool:
        return [], []
    q = c.vec[x.sample_id]
    key = (lambda i: c.vec[i['sample_id']]) if c.cfg['retrieval'].get('key') == 'question' else \
        (lambda i: wtu_vector(c, i['when_to_use']))
    scored = sorted(((float(key(i) @ q), i) for i in pool), key=lambda t: (-t[0], t[1]['id']))
    s1 = scored[0][0]
    s2 = scored[1][0] if len(scored) > 1 else None
    rule = c.rule[b]
    ranking = [{'id': i['id'], 'version': i['version'], 'score': round(s, 4)} for s, i in scored[:3]]
    if rule.get('mode') == 'coverage':  # no threshold yet (too little history): show nothing
        ok = rule['threshold'] is not None and s1 >= rule['threshold']
        return ([i for _, i in scored[:c.cfg['retrieval']['max_examples']]] if ok else []), ranking
    if rule.get('mode') == 'absolute':
        return ([scored[0][1]] if s1 >= rule['threshold'] else []), ranking
    ok = s1 >= rule['threshold'] and (s2 is None or s1 - s2 >= rule['margin'])
    return ([scored[0][1]] if ok else []), ranking


def update_rule(c, b):
    rule = c.rule[b]
    if rule.get('mode') == 'coverage':
        tops = c.past_top[b]
        rule['threshold'] = (float(np.quantile(tops, 1 - rule['coverage'])) if len(tops) >= c.cfg['retrieval']['min_history']
                             else None)


def ensure_vanilla(c, ids):
    """New benchmark without a vanilla run: answer the questions without samples first (cached by input)."""
    todo = [c.samples[i] for i in ids if i not in c.vanilla]
    if not todo:
        return
    res = gen(c, [{'sample': x, 'parts': [VIDEO, user_prompt(x)]} for x in todo])
    for x, r in zip(todo, res):
        c.vanilla[x.sample_id] = {'raw': r['raw'], 'answer': r['answer'], 'valid': r['valid'], 'correct': r['correct'],
                                  'user': user_prompt(x), 'system': c.system}


# ------------------------------------------------------------------ online block
def answer_block(c, b, k, mem):
    plans, meta = [], []
    items = mem.listed() if c.cfg['retrieval'].get('pool') == 'listed' else mem.active()  # listed: active + held
    ensure_vanilla(c, c.blocks[b][k - 1])
    update_rule(c, b)
    for sid in c.blocks[b][k - 1]:
        x = c.samples[sid]
        shown, ranking = retrieve(c, b, x, items)
        parts = parts_for(c, x, shown) if shown else None
        plans.append({'sample': x, 'parts': parts})
        meta.append({'shown': shown, 'ranking': ranking, 'parts': parts})
    results = gen(c, plans)
    rows = []
    for j, (p, m, r) in enumerate(zip(plans, meta, results), 1):
        x, v = p['sample'], c.vanilla[p['sample'].sample_id]
        scores = {e['id']: e['score'] for e in m['ranking']}
        rows.append({'benchmark': b, 'block': k, 'index': (k - 1) * c.cfg['block_size'] + j, 'sample_id': x.sample_id,
                     'video': c.video[x.sample_id], 'never_used_question': c.never_used.get(x.sample_id),
                     'shown': [{'id': i['id'], 'version': i['version'], 'kind': i['kind'], 'score': scores.get(i['id'])}
                               for i in m['shown']],
                     'ranking': m['ranking'], 'rule': dict(c.rule[b]),
                     'student_input_text': [t for t in (m['parts'] or []) if t != VIDEO] or None,
                     'student': {'answer': r['answer'], 'correct': r['correct'], 'valid': r['valid'], 'output': r['raw'],
                                 'source': r['source']},
                     'vanilla': {'answer': v['answer'], 'correct': v['correct'], 'valid': v['valid'], 'output': v['raw']},
                     'correct': r['correct'], 'vanilla_correct': v['correct'],
                     'shown_answer_text_equals_gt_text': any(norm(i['choices'][i['ground_truth']]) == norm(x.choices[x.ground_truth])
                                                             for i in m['shown'])})
    c.past_top[b] += [r['ranking'][0]['score'] for r in rows if r['ranking']]
    return rows


def case_text(c, cid, row, with_answer=True):
    x = c.samples[row['sample_id']]
    s, v = row['student'], row['vanilla']
    change = 'gain' if row['correct'] and not row['vanilla_correct'] else 'harm' if row['vanilla_correct'] and not row['correct'] else 'same'
    shown = ', '.join(f'{i["id"]} v{i["version"]} (score {i["score"]})' for i in row['shown']) or 'none'
    head = f'=== case {cid} ===\nquestion:\n{x.question}\noptions:\n{options_text(x.choices)}\n'
    if with_answer:
        head += (f'correct option: {x.ground_truth}\nmodel answer: {s["answer"] or "invalid"} ({"right" if s["correct"] else "wrong"}); '
                 f'without the example: {v["answer"] or "invalid"} ({"right" if v["correct"] else "wrong"}); change: {change}\n')
    return head + f'example shown: {shown}\nmodel output:\n{s["output"].strip()}'


# ------------------------------------------------------------------ stage 1
def stage1(c, b, k, mem, rows):
    cases = {f'Q{j:02d}': r for j, r in enumerate(rows, 1)}
    shown_ids = sorted({i['id'] for r in rows for i in r['shown']})
    index = '\n'.join(f'- {i["id"]} ({i["kind"]}, {i["status"]}): {i["when_to_use"]}' for i in mem.listed()) or '(empty)'
    text = (f'BENCHMARK: {c.cfg["benchmarks"][b]}  BLOCK {k}\nMEMORY INDEX ({len(mem.active())} active):\n{index}\n\n'
            'ITEMS SHOWN IN THIS BLOCK:\n' + ('\n\n'.join(render_item(mem.items[i]) for i in shown_ids) or '(none)')
            + '\n\nCASES:\n\n' + '\n\n'.join(case_text(c, cid, r) for cid, r in cases.items()))
    prompt = STAGE1_PROMPT.replace('{max_selected}', str(c.cfg['stage1']['max_selected']))
    out = call(c, 'stage1_select', prompt, [('text', text)], Stage1Output)
    picked = []
    for s in sorted(out.selected_cases, key=lambda s: s.priority):
        if s.case_id in cases and s.case_id not in [p.case_id for p in picked]:
            picked.append(s)
    picked = picked[:c.cfg['stage1']['max_selected']]
    return cases, picked, {'input_chars': len(text), 'output': out.model_dump(), 'used': [p.case_id for p in picked]}


# ------------------------------------------------------------------ stage 2 (blind)
def stage2(c, views):
    """views: teacher_view() dicts only (frames, question, options). No answers, outputs or stage-1 reasons."""
    from src.datasets.video_loader import load_frames
    for v in views:
        if set(v) != {'frame_paths', 'question', 'choices'}:
            raise RuntimeError(f'stage 2 got fields beyond frames/question/options: {sorted(v)}')
    ids = [f'V{n + 1}' for n in range(len(views))]
    blocks = [('text', 'Cases follow, each with a case_id, the question, options and 32 frames.')]
    for cid, v in zip(ids, views):
        frames, meta = load_frames(v['frame_paths'], **c.cfg['student']['video'])
        blocks.append(('text', f'\n=== case_id {cid} ===\nQUESTION:\n{v["question"]}\nOPTIONS:\n{options_text(v["choices"])}\nFRAMES:'))
        blocks += [('image', label, url) for label, url in frame_payload(frames, meta['sampled_fps'])]
    out = call(c, 'stage2_solve', STAGE2_PROMPT, blocks, Stage2Output)
    got = {s.case_id: s.model_dump() for s in out.solutions}
    return [got.get(cid) for cid in ids], {'case_ids': ids, 'text_blocks': [b[1] for b in blocks if b[0] == 'text']}


def solution_text(sol, truth):
    if not sol:
        return 'no solution returned'
    ev = '\n'.join(f'  - [{e["frames"]}] ({e["basis"]}) {e["observation"]}' for e in sol['evidence'])
    unc = '; '.join(sol['uncertainties']) or 'none'
    return (f'answer {sol["answer"]} ({"right" if sol["answer"].strip().upper() == truth else "wrong"}), confidence {sol["confidence"]}\n'
            f'evidence:\n{ev}\nrelation: {sol["answer_relation"]}\nuncertainties: {unc}')


# ------------------------------------------------------------------ stage 3
def nearest_items(c, mem, sample_id, n=3):
    """The n memory items closest to a case, by the retrieval key (when_to_use, or the item's source question)."""
    q = c.vec[sample_id]
    key = (lambda i: c.vec[i['sample_id']]) if c.cfg['retrieval'].get('key') == 'question' else \
        (lambda i: wtu_vector(c, i['when_to_use']))
    return sorted(mem.listed(), key=lambda i: -float(key(i) @ q))[:n]


def stage3(c, b, k, mem, cases, picked, sols):
    related = {}
    for p in picked:
        r = cases[p.case_id]
        for i in [mem.items[s['id']] for s in r['shown']] + nearest_items(c, mem, r['sample_id']):
            related[i['id']] = i
    text = ('RELATED MEMORY ITEMS:\n' + ('\n\n'.join(render_item(i) for i in related.values()) or '(none)') + '\n\nCASES:\n\n'
            + '\n\n'.join(case_text(c, p.case_id, cases[p.case_id]) + f'\nselection reason: {p.selection_reason}\n'
                          f'teacher solution (independent, blind):\n{solution_text(sols[p.case_id], c.samples[cases[p.case_id]["sample_id"]].ground_truth)}'
                          for p in picked))
    out = call(c, 'stage3_compare', STAGE3_PROMPT, [('text', text)], Stage3Output)
    cands, rejected = {}, []
    for n, cand in enumerate(out.candidates, 1):
        why, src = None, cases.get(cand.source_case_ids[0]) if cand.source_case_ids else None
        sol = sols.get(cand.source_case_ids[0]) if cand.source_case_ids else None
        if not src or cand.source_case_ids[0] not in sols:
            why = 'first source case is not a selected case'
        elif not sol or sol['answer'].strip().upper() != c.samples[src['sample_id']].ground_truth:
            why = 'teacher did not answer the source case correctly'
        elif cand.kind == 'success' and not src['correct']:
            why = 'success candidate on a case the student got wrong'
        elif cand.kind == 'corrected' and src['correct']:
            why = 'corrected candidate on a case the student got right'
        x = c.samples[src['sample_id']] if src else None
        example = answers(cand.worked_output, x.choices, x.ground_truth) if x else None
        if not why and not example:
            why = 'worked_output does not answer the correct option of its question'
        if why:
            rejected.append({**cand.model_dump(), 'rejected_because': why})
            continue
        cid = f'cand-{k:03d}-{n}'
        cands[cid] = {'candidate_id': cid, 'kind': cand.kind, 'sample_id': x.sample_id, 'video': c.video[x.sample_id],
                      'question': x.question, 'choices': x.choices, 'ground_truth': x.ground_truth,  # the real question
                      'source_sample_ids': [cases[s]['sample_id'] for s in cand.source_case_ids if s in cases],
                      'worked_output': example, 'supported_facts': cand.supported_facts,
                      'transferable_operation': cand.transferable_operation, 'risks': cand.risks,
                      'when_to_use': '', 'strategy': '', 'rounds': [], 'trial_summary': [], 'next_action': None}
    return cands, {'input_chars': len(text), 'output': out.model_dump(), 'rejected': rejected, 'related': list(related)}


# ------------------------------------------------------------------ stage 4
def past_rows(c, b, k):
    rows = []
    for j in range(1, k + 1):
        p = block_path(c, 'rows', b, j)
        rows += json.loads(p.read_text()) if p.exists() else []
    return rows


def pick_trials(c, cand, history, used, n, min_right):
    """Answered questions from other videos, most similar to the source question first (WeMM question vectors), with
    at least min_right questions the student answered correctly without examples (regression checks)."""
    q = c.vec[cand['sample_id']]
    probe = {'id': cand['candidate_id'], 'video': cand['video'], 'question': cand['question'], 'choices': cand['choices'],
             'ground_truth': cand['ground_truth']}
    pool = [r for r in history if r['sample_id'] not in used and eligible(c, c.samples[r['sample_id']], [probe])]
    pool.sort(key=lambda r: (-float(c.vec[r['sample_id']] @ q), r['sample_id']))
    chosen = pool[:n]
    right = [r for r in chosen if r['vanilla_correct']]
    for r in [r for r in pool[n:] if r['vanilla_correct']][:max(0, min_right - len(right))]:
        drop = next((x for x in reversed(chosen) if not x['vanilla_correct']), None)
        if drop is None:
            break
        chosen[chosen.index(drop)] = r
    return chosen


def run_trials(c, cand, trials, rnd):
    item = {'id': cand['candidate_id'], 'question': cand['question'], 'choices': cand['choices'],
            'ground_truth': cand['ground_truth'], 'worked_output': cand['worked_output']}
    plans = [{'sample': c.samples[r['sample_id']], 'parts': parts_for(c, c.samples[r['sample_id']], item, f'@r{rnd}')}
             for r in trials]
    plans.insert(0, {'sample': c.samples[cand['sample_id']], 'parts': None})  # source question: development diagnostic
    src = c.samples[cand['sample_id']]
    plans[0]['parts'] = parts_for(c, src, item, f'@r{rnd}')
    res = gen(c, plans)
    out = []
    for p, r in zip(plans, res):
        x, v = p['sample'], c.vanilla[p['sample'].sample_id]
        out.append({'sample_id': x.sample_id, 'role': 'source' if x.sample_id == cand['sample_id'] else 'other_video',
                    'answer': r['answer'], 'correct': r['correct'], 'valid': r['valid'], 'output': r['raw'],
                    'vanilla_answer': v['answer'], 'vanilla_correct': v['correct'],
                    'student_input_text': [t for t in p['parts'] if t != VIDEO]})
    return out


def trial_text(c, t):
    x = c.samples[t['sample_id']]
    return (f'--- trial on {"the source question itself (development diagnostic)" if t["role"] == "source" else "a question from another video"} ---\n'
            f'question:\n{x.question}\noptions:\n{options_text(x.choices)}\ncorrect option: {x.ground_truth}\n'
            f'with the example: {t["answer"] or "invalid"} ({"right" if t["correct"] else "wrong"}); without: '
            f'{t["vanilla_answer"] or "invalid"} ({"right" if t["vanilla_correct"] else "wrong"})\n'
            f'student output with the example (example options were relabelled for this question):\n{t["output"].strip()}')


def cand_text(c, cand, with_trials):
    text = (f'=== candidate {cand["candidate_id"]} ({cand["kind"]}) ===\nquestion:\n{cand["question"]}\noptions:\n'
            f'{options_text(cand["choices"])}\ncorrect option: {cand["ground_truth"]}\nworked_output:\n{cand["worked_output"]}\n'
            f'supported facts: {"; ".join(cand["supported_facts"])}\ntransferable operation: {cand["transferable_operation"]}\n'
            f'risks: {cand["risks"]}\n')
    if cand['when_to_use']:
        text += f'current when_to_use: {cand["when_to_use"]}\ncurrent strategy: {cand["strategy"]}\n'
    if with_trials and cand['rounds'] and cand['rounds'][-1].get('trials'):
        text += 'STUDENT TRIALS OF THE CURRENT TEXT:\n' + '\n'.join(trial_text(c, t) for t in cand['rounds'][-1]['trials'])
    return text


def stage4(c, b, k, cands, rows):
    if not cands:
        return {}, {'rounds': []}
    cfg, history, record = c.cfg['stage4'], past_rows(c, b, k - 1) + rows, {'rounds': []}
    prompt = STAGE4_PROMPT.replace('{position}', 'after the video and before the new question'
                                   if c.cfg['retrieval']['position'] == 'after_video' else 'before the video')
    open_ids = list(cands)
    for rnd in range(cfg['max_rounds'] + 1):
        if not open_ids:
            break
        text = '\n\n'.join(cand_text(c, cands[i], rnd > 0) for i in open_ids)
        out = call(c, 'stage4_refine', prompt, [('text', text)], Stage4Output)
        got = {o.candidate_id: o for o in out.candidates}
        rec = {'round': rnd, 'input_chars': len(text), 'output': out.model_dump(), 'invalid': []}
        nxt = []
        for i in open_ids:
            cand, o = cands[i], got.get(i)
            if o is None:
                rec['invalid'].append({'candidate_id': i, 'why': 'no revision returned'})
                cand['next_action'] = cand['next_action'] or 'hold'
                continue
            example = answers(o.worked_output, cand['choices'], cand['ground_truth'])
            if not example:
                rec['invalid'].append({'candidate_id': i, 'why': 'revised worked_output does not answer the correct option'})
            else:
                cand['worked_output'] = example
            cand.update(when_to_use=o.when_to_use.strip() or cand['when_to_use'], strategy=o.strategy.strip() or cand['strategy'],
                        next_action=o.next_action)
            cand['rounds'].append({'round': rnd, 'when_to_use': cand['when_to_use'], 'strategy': cand['strategy'],
                                   'worked_output': cand['worked_output'], 'changes': o.changes, 'trial_review': o.trial_review,
                                   'next_action': o.next_action})
            if rnd < cfg['max_rounds'] and o.next_action in ('more_trial', 'revise') or (rnd == 0 and o.next_action == 'memory_review'):
                nxt.append(i)
        record['rounds'].append(rec)
        if rnd == cfg['max_rounds']:
            break
        for i in nxt:  # student trials of the current text on past questions of other videos
            cand = cands[i]
            used = {t['sample_id'] for r in cand['rounds'] for t in r.get('trials', [])}
            trials = pick_trials(c, cand, history, used, cfg['trial_questions'], cfg['min_vanilla_correct'])
            res = run_trials(c, cand, trials, rnd)
            cand['rounds'][-1]['trials'] = res
            other = [t for t in res if t['role'] == 'other_video']
            cand['trial_summary'].append({'round': rnd, 'text_digest': digest(cand['worked_output'])[:12],
                                          'source_correct': res[0]['correct'], 'other_n': len(other),
                                          'other_correct': sum(t['correct'] for t in other),
                                          'other_vanilla_correct': sum(t['vanilla_correct'] for t in other),
                                          'gains': sum(t['correct'] and not t['vanilla_correct'] for t in other),
                                          'harms': sum(t['vanilla_correct'] and not t['correct'] for t in other)})
        open_ids = nxt
    return cands, record


# ------------------------------------------------------------------ stage 5
def stage5(c, b, k, mem, cands, rows):
    live = {i: x for i, x in cands.items() if x['next_action'] != 'discard'}
    shown = Counter(s['id'] for r in rows for s in r['shown'])
    if not live and not shown:
        return [], [], {'skipped': 'no candidates and no item used in this block'}
    w = [r for r in rows if r['shown']]
    online = (f'this block: {len(w)} of {len(rows)} questions got an example; gains {sum(r["correct"] and not r["vanilla_correct"] for r in w)}, '
              f'harms {sum(r["vanilla_correct"] and not r["correct"] for r in w)}; items used: '
              + (', '.join(f'{i} x{n}' for i, n in shown.items()) or 'none'))
    cand_part = '\n\n'.join(cand_text(c, x, False) + 'trial summary: ' + json.dumps(x['trial_summary']) + '\nlast review: '
                            + (x['rounds'][-1]['trial_review'] if x['rounds'] else '') + f'\nstage-4 next_action: {x["next_action"]}'
                            for x in live.values()) or '(none)'
    text = (f'BENCHMARK: {c.cfg["benchmarks"][b]}  BLOCK {k}\nONLINE USAGE: {online}\n\nMEMORY ({len(mem.active())} active, '
            f'{len(mem.listed()) - len(mem.active())} held):\n' + ('\n\n'.join(render_item(i) for i in mem.listed()) or '(empty)')
            + '\n\nNEW CANDIDATES:\n\n' + cand_part)
    prompt = STAGE5_PROMPT.replace('{max_active}', str(c.cfg['memory']['max_active']))
    out = call(c, 'stage5_manage', prompt, [('text', text)], Stage5Output)
    applied, rejected = mem.apply(out, live, k)
    return applied, rejected, {'input_chars': len(text), 'output': out.model_dump()}


# ------------------------------------------------------------------ block and benchmark loops
def do_block(c, b, k, mem):
    rows = answer_block(c, b, k, mem)
    mem.record_usage(rows)
    record = {'benchmark': b, 'block': k, 'memory_before': [(i['id'], i['version'], i['status']) for i in mem.listed()]}
    if k < len(c.blocks[b]):  # the last block's lessons could never be used
        cases, picked, record['stage1'] = stage1(c, b, k, mem, rows)
        views = [teacher_view(c.samples[cases[p.case_id]['sample_id']]) for p in picked]
        sols_list, record['stage2'] = stage2(c, views) if views else ([], {})
        sols = {p.case_id: s for p, s in zip(picked, sols_list)}
        record['stage2']['solutions'] = dict(sols)
        cands, record['stage3'] = stage3(c, b, k, mem, cases, picked, sols) if picked else ({}, {})
        cands, record['stage4'] = stage4(c, b, k, cands, rows)
        record['candidates'] = cands
        applied, rejected, record['stage5'] = stage5(c, b, k, mem, cands, rows)
        record['stage5'].update(applied=applied, rejected=rejected)
    record['memory_after'] = [(i['id'], i['version'], i['status']) for i in mem.listed()]
    atomic_json(block_path(c, 'stages', b, k), json.loads(json.dumps(record, default=str)))
    atomic_json(block_path(c, 'memory', b, k), mem.snapshot())
    atomic_json(block_path(c, 'rows', b, k), rows)  # last: marks the block complete
    return rows, record


def run_benchmark(c, b, max_blocks):
    k = 0
    while block_path(c, 'rows', b, k + 1).exists():
        k += 1
    mem = StageMemory(b, json.loads(block_path(c, 'memory', b, k).read_text()) if k else [], c.cfg['memory']['max_active'])
    c.past_top[b] = [r['ranking'][0]['score'] for j in range(1, k + 1)
                     for r in json.loads(block_path(c, 'rows', b, j).read_text()) if r['ranking']]
    if not c.vanilla or any(i not in c.vanilla for i in c.order[b][:k * c.cfg['block_size']]):
        ensure_vanilla(c, c.order[b][:k * c.cfg['block_size']])  # resume: cached generations, needed for trials
    last = min(len(c.blocks[b]), max_blocks or len(c.blocks[b]))
    log(c, f'{b}: resume after block {k} of {last}; active items {len(mem.active())}')
    current = {'block': k + 1}
    if not c.cfg.get('vanilla'):  # vanilla answers do not depend on the memory: fill GPU time while the API works
        def prefetch():  # at most 3 blocks ahead, yielding the GPU lock between blocks
            for kk in range(k + 1, last + 1):
                while kk > current['block'] + 3:
                    time.sleep(5)
                ensure_vanilla(c, c.blocks[b][kk - 1])
                time.sleep(2)
        threading.Thread(target=prefetch, daemon=True).start()
    for kk in range(k + 1, last + 1):
        current['block'] = kk
        t0 = time.time()
        rows, rec = do_block(c, b, kk, mem)
        w = [r for r in rows if r['shown']]
        ops = Counter(a['op'] for a in rec.get('stage5', {}).get('applied', []))
        log(c, f'{b} block {kk}: icl {sum(r["correct"] for r in rows)}/{len(rows)} vanilla {sum(r["vanilla_correct"] for r in rows)} | '
            f'with example {len(w)} (+{sum(r["correct"] and not r["vanilla_correct"] for r in w)}/-'
            f'{sum(r["vanilla_correct"] and not r["correct"] for r in w)}) | selected {len(rec.get("stage1", {}).get("used", []))} '
            f'candidates {len(rec.get("candidates", {}))} ops {dict(ops)} | active {len(mem.active())} | '
            f'{(time.time() - t0) / 60:.1f} min, teacher calls {c.teacher.used}')
        export(c)


# ------------------------------------------------------------------ export
def write_jsonl(path, header, rows):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        f.write(json.dumps(header, ensure_ascii=False) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def paired(sub):
    st = paired_stats([r['vanilla_correct'] for r in sub], [r['correct'] for r in sub], [r['video'] for r in sub])
    return {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v] if isinstance(v, list) else v) for k, v in st.items()}


def export(c):
    with c.log_lock:
        _export(c)


def _export(c):
    overall, all_rows = [], []
    for b in c.blocks:
        rows, recs, k = [], [], 1
        while block_path(c, 'rows', b, k).exists():
            rows += json.loads(block_path(c, 'rows', b, k).read_text())
            recs.append(json.loads(block_path(c, 'stages', b, k).read_text()))
            k += 1
        if not rows:
            continue
        mem = json.loads(block_path(c, 'memory', b, k - 1).read_text())
        w = [r for r in rows if r['shown']]
        acc = lambda xs, key: round(sum(r[key] for r in xs) / len(xs), 4) if xs else None
        trials = [dict(t, block=rec['block'], candidate_id=cid, round=rd['round'])
                  for rec in recs for cid, cand in rec.get('candidates', {}).items() for rd in cand['rounds'] for t in rd.get('trials', [])]
        other = [t for t in trials if t['role'] == 'other_video']
        s = {'benchmark': c.cfg['benchmarks'][b], 'total_questions': len(c.order[b]), 'questions': len(rows), 'blocks': k - 1,
             'accuracy': {'icl': {'accuracy': acc(rows, 'correct'), 'invalid': sum(not r['student']['valid'] for r in rows)},
                          'vanilla': {'accuracy': acc(rows, 'vanilla_correct'), 'invalid': sum(not r['vanilla']['valid'] for r in rows)}},
             'icl_vs_vanilla': paired(rows), 'with_example': {'n': len(w), 'icl_vs_vanilla': paired(w)},
             'with_example_answer_text_equals_gt': {'n': sum(r['shown_answer_text_equals_gt_text'] for r in w),
                                                    'icl_vs_vanilla': paired([r for r in w if r['shown_answer_text_equals_gt_text']])},
             'never_used_questions': paired([r for r in rows if r['never_used_question']]),
             'development_trials': {'n_other_video': len(other), 'with_example_accuracy': acc(other, 'correct'),
                                    'vanilla_accuracy': acc(other, 'vanilla_correct'),
                                    'source_question_accuracy': acc([t for t in trials if t['role'] == 'source'], 'correct'),
                                    'note': 'past questions re-answered to develop candidates; not online performance'},
             'memory': dict(Counter(i['status'] for i in mem)),
             'rule': 'icl = first answer with the memory frozen before the block; vanilla = same student, same question, no example'}
        write_jsonl(c.out / f'model_output_{b}.jsonl', {'_type': 'summary', **s}, rows)
        write_jsonl(c.out / f'ics_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b], 'after_block': k - 1,
                    'status': dict(Counter(i['status'] for i in mem)), 'accuracy': s['accuracy'], 'with_example': s['with_example']}, mem)
        write_jsonl(c.out / f'stages_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b], 'blocks': k - 1,
                    'accuracy': s['accuracy']}, recs)
        write_jsonl(c.out / f'trials_{b}.jsonl', {'_type': 'summary', 'benchmark': c.cfg['benchmarks'][b], **s['development_trials'],
                    'accuracy': {'with_example': s['development_trials']['with_example_accuracy'],
                                 'vanilla': s['development_trials']['vanilla_accuracy']}}, trials)
        overall.append(s)
        all_rows += rows
    if not all_rows:
        return
    usage = c.teacher.usage()
    import sqlite3
    gens = sqlite3.connect(str(c.root / 'predictions.sqlite'), timeout=60).execute(
        'SELECT COUNT(*), SUM(input_tokens), SUM(generated_tokens) FROM p').fetchone()
    w = [r for r in all_rows if r['shown']]
    write_jsonl(c.out / 'overall_accuracy.jsonl', {
        '_type': 'summary', 'benchmark': 'all', 'updated_at': now(), 'questions': len(all_rows),
        'accuracy': {'icl': round(sum(r['correct'] for r in all_rows) / len(all_rows), 4),
                     'vanilla': round(sum(r['vanilla_correct'] for r in all_rows) / len(all_rows), 4)},
        'icl_vs_vanilla': paired(all_rows), 'with_example': {'n': len(w), 'icl_vs_vanilla': paired(w)},
        'cost': {'teacher_calls_reserved': c.teacher.used, 'by_stage': usage,
                 'input_tokens': sum(v['input_tokens'] for v in usage.values()),
                 'output_tokens': sum(v['output_tokens'] for v in usage.values()),
                 'student_generations': gens[0], 'student_input_tokens': gens[1], 'student_generated_tokens': gens[2]},
        'retrieval_rule': c.rule}, overall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/icl_stages.yaml')
    ap.add_argument('--max-blocks', type=int)
    ap.add_argument('--benchmarks')
    ap.add_argument('--max-calls', type=int, help='raise the teacher/critic call cap for this run (logged)')
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--gpus', help='override student GPUs, e.g. 2,3')
    a = ap.parse_args()
    if a.export_only:
        export(setup(a.config, gpu=False))
        return
    c = setup(a.config, gpu=True, gpus=[int(g) for g in a.gpus.split(',')] if a.gpus else None)
    if a.max_calls:
        log(c, f'teacher call cap {c.teacher.max_calls} -> {a.max_calls} (command line)')
        c.teacher.max_calls = a.max_calls
    if a.max_blocks:
        log(c, f'max blocks per benchmark {c.cfg["max_blocks"]} -> {a.max_blocks} (command line)')
    names = a.benchmarks.split(',') if a.benchmarks else list(c.blocks)
    max_blocks = a.max_blocks or c.cfg['max_blocks']
    try:
        with ThreadPoolExecutor(len(names)) as ex:
            futures = {b: ex.submit(run_benchmark, c, b, max_blocks) for b in names}
            for b, f in futures.items():
                try:
                    f.result()
                except FATAL as exc:
                    log(c, f'{b}: stopped ({type(exc).__name__}: {exc})')
                    raise
    finally:
        c.pool.close()
        log(c, 'exit')


if __name__ == '__main__':
    main()
