"""p1 of every vanilla answer: the student's probability of its answer letter, renormalised over the question's options.

For each question of the config's benchmarks, the vanilla output is cut where its answer letter starts (right after
'<ANSWER', before the '>'); one forward pass on the vanilla input + that prefix gives the next-token probabilities of '>A', '>B', ...
for the question's own option letters, renormalised to sum to 1; p1 is the largest. Greedy decoding picks that same
letter, so p1 is what the student would report while generating. An output without an option letter right after
'<ANSWER' has no p1 (stored as NULL): run_icl_confidence treats it as low confidence.

Output: the confidence.p1_db of the config (table a: sample_id, p1, argmax, vanilla_answer, probs, source, created_at).
Questions already stored are skipped, so the script resumes.

    fuzzy35/bin/python scripts/compute_p1.py --gpus 0,1,2,3 [--config configs/final_c.yaml] [--db path] [--limit n]
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_icl_confidence as RC  # noqa: E402
from src.icl.prompts import VIDEO, user_prompt  # noqa: E402
from src.utils.storage import now  # noqa: E402

ANSWER = re.compile(r'<ANSWER>', re.I)
BATCH = 64


def open_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), isolation_level=None, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS a (sample_id TEXT PRIMARY KEY, p1 REAL, argmax TEXT, vanilla_answer TEXT, '
               'probs TEXT, source TEXT, created_at TEXT)')
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/final_c.yaml')
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--db', help='output database (default: confidence.p1_db of the config)')
    ap.add_argument('--limit', type=int, help='score at most this many new questions (a quick check)')
    a = ap.parse_args()
    c = RC.setup(gpu=False, config=a.config, check_p1=False)
    db = open_db(a.db or c.cf['p1_db'])
    done = {s for (s,) in db.execute('SELECT sample_id FROM a')}
    tasks, kw, sids = [], [], []
    for t in c.tracks:
        for s in c.order[t]:
            if s in done:
                continue
            x, v = c.samples[s], c.vanilla[s]
            raw = v['raw'] or ''
            m = ANSWER.search(raw)
            if not m or raw[m.end():m.end() + 1] not in x.choices:  # no tag, or the letter not right after it
                db.execute('INSERT INTO a VALUES (?,?,?,?,?,?,?)', (s, None, 'unscored' if m else None, v['answer'], None,
                                                                    'not scored', now()))
                continue
            tasks.append((x.model_input(), RC.R.seed_of(c, s), c.system))
            kw.append({'parts': [VIDEO, user_prompt(x)], 'prefix': raw[:m.end() - 1], 'letters': tuple(sorted(x.choices)),
                       'token_prefix': '>'})
            sids.append(s)
    if a.limit is not None:
        tasks, kw, sids = tasks[:a.limit], kw[:a.limit], sids[:a.limit]
    print(f'{len(tasks)} to score, {len(done)} stored', flush=True)
    if not tasks:
        return
    from src.icl.pool import BackbonePool
    st = c.cfg['student']
    pool = BackbonePool([int(g) for g in a.gpus.split(',')], st['backbone'], st['video'], st['generation'])
    ref = next(iter(c.vanilla.values()))['identity']
    diff = sorted(k for k in set(ref) | set(pool.runtime_identity) if k != 'device' and ref.get(k) != pool.runtime_identity.get(k))
    if diff:
        pool.close()
        raise RuntimeError(f'student runtime differs from the vanilla run in {diff}')
    agree = 0
    try:
        for i in range(0, len(tasks), BATCH):
            t0 = time.time()
            gens = pool.generate_many(tasks[i:i + BATCH], method='answer_scores', kwargs=kw[i:i + BATCH])
            for s, g in zip(sids[i:i + BATCH], gens):
                r = json.loads(g.raw_output)
                va = c.vanilla[s]['answer']
                agree += r['argmax'] == va
                db.execute('INSERT OR REPLACE INTO a VALUES (?,?,?,?,?,?,?)',
                           (s, max(r['probs'].values()), r['argmax'], va, json.dumps(r['probs']), 'forward', now()))
            print(f'{min(i + BATCH, len(tasks))}/{len(tasks)} in {time.time() - t0:.0f}s; top letter = vanilla answer '
                  f'{agree}/{min(i + BATCH, len(tasks))}', flush=True)
    finally:
        pool.close()


if __name__ == '__main__':
    main()
