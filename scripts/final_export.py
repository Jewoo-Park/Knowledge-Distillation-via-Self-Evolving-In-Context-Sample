"""Per-benchmark results of the final confidence-gated run (configs/final_c.yaml), written while it runs.

results_dir/<benchmark>/ is rewritten every export.every_blocks blocks and when the benchmark ends:
  model_output.jsonl  one row per answered question (summary line first)
  memory.jsonl        the memory after the last answered block, inactive items included (summary line first)
  block_curve.jsonl   accuracy per block, ICS vs vanilla, memory size, teacher spend (summary line first)
  summary.json, summary.md   ICS vs vanilla: video-group bootstrap CI, fixes / breaks, sign test, by gate and p1 band
results_dir/overall.jsonl collects the summary lines of all benchmarks.

    fuzzy35/bin/python scripts/final_export.py [--config configs/final_c.yaml]   # rewrite from the run root, no GPU
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.icl.stats import paired_stats  # noqa: E402
from src.utils.storage import atomic_json, now  # noqa: E402

BANDS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 0.999), (0.999, 1.01)]


def load_rows(c, t):
    rows, k = [], 1
    while (c.root / 'rows' / t / f'block_{k:04d}.json').exists():
        rows += json.loads((c.root / 'rows' / t / f'block_{k:04d}.json').read_text())
        k += 1
    return rows, k - 1


def write_jsonl(path, head, rows):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for r in [head] + rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def acc(rows, key):
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else None


def part(rows):
    if not rows:
        return {'questions': 0}
    fix = sum(r['correct'] and not r['vanilla_correct'] for r in rows)
    brk = sum(r['vanilla_correct'] and not r['correct'] for r in rows)
    return {'questions': len(rows), 'ics_accuracy': acc(rows, 'correct'), 'vanilla_accuracy': acc(rows, 'vanilla_correct'),
            'fixes': fix, 'breaks': brk, 'net_pp': round((fix - brk) / len(rows) * 100, 2)}


def same_answer_text(c, t, rows):
    """Shown examples whose correct option text equals the current question's correct option text (label used only
    for this report; exclude_option_text already drops items whose answer text is among the current options)."""
    snaps, n = {}, 0
    for r in rows:
        if not r['shown']:
            continue
        k = r['block'] - 1
        if k not in snaps:
            p = c.root / 'memory' / t / f'block_{k:04d}.json'
            snaps[k] = {(i['id'], i['version']): i for i in json.loads(p.read_text())} if p.exists() else {}
        item = snaps[k].get((r['shown'][0]['id'], r['shown'][0]['version']))
        x = c.samples[r['sample_id']]
        if item and item['choices'].get(item['ground_truth'], '').strip().lower() == x.choices[x.ground_truth].strip().lower():
            n += 1
    return n


def export(c, t, finished=False):
    rows, last = load_rows(c, t)
    if not rows:
        return None
    out = Path(c.cfg['results_dir']) / t
    out.mkdir(parents=True, exist_ok=True)
    total = len(c.blocks[t])
    st = paired_stats([r['vanilla_correct'] for r in rows], [r['correct'] for r in rows], [r['group'] for r in rows])
    gated = [r for r in rows if r['gated']]
    shown = [r for r in rows if r['tried']]
    stage = json.loads((c.root / 'stages' / t / f'block_{last:04d}.json').read_text())
    costs = stage.get('costs', {})
    bands = {}
    for lo, hi in BANDS:
        b = [r for r in rows if r.get('p1') is not None and lo <= r['p1'] < hi]
        bands[f'[{lo}, {min(hi, 1.0)}{"]" if hi > 1 else ")"}'] = dict(part(b), example_runs=sum(bool(r['tried']) for r in b))
    bands['p1 unreadable'] = dict(part([r for r in rows if r.get('p1') is None]),
                                  example_runs=sum(bool(r['tried']) for r in rows if r.get('p1') is None))
    head = {'summary': True, 'benchmark': t, 'status': 'finished' if finished else f'running (block {last} of {total})',
            'made_at': now(), 'questions': len(rows), 'blocks': f'{last}/{total}',
            'ics_accuracy': acc(rows, 'correct'), 'vanilla_accuracy': acc(rows, 'vanilla_correct'),
            'difference_pp': round(st['delta_pp'], 2), 'ci95_pp': [round(v, 2) for v in st['ci95_pp']],
            'fixes': st['gains'], 'breaks': st['harms'], 'sign_test_p': st['sign_test_p'],
            'gated': len(gated), 'example_runs': len(shown), 'gated_without_item': len(gated) - len(shown),
            'invalid_example_output_kept_vanilla': sum(r['fallback'] for r in rows),
            'invalid_final': sum(not r['student']['valid'] for r in rows),
            'invalid_vanilla': sum(not r['vanilla']['valid'] for r in rows),
            'shown_example_with_same_answer_text': same_answer_text(c, t, shown),
            'teacher_usd': costs.get('teacher_usd'), 'teacher_calls': costs.get('teacher_calls'),
            'rule': f'example (top-1 by source-question cosine, no similarity floor) when vanilla p1 < '
                    f'{c.cf["p1_threshold"]} or p1 unreadable; invalid example output keeps the vanilla answer'}
    detail = {'with_example': part(shown), 'gated_no_item': part([r for r in gated if not r['tried']]),
              'not_gated': part([r for r in rows if not r['gated']]), 'p1_bands': bands}
    write_jsonl(out / 'model_output.jsonl', head, [
        {k: r.get(k) for k in ('index', 'block', 'sample_id', 'group', 'p1', 'gated', 'selection', 'shown', 'icl_answer',
                               'icl_valid', 'icl_correct', 'fallback', 'student', 'vanilla', 'correct', 'vanilla_correct',
                               'student_input_text')} for r in rows])
    mp = c.root / 'memory' / t / f'block_{last:04d}.json'
    items = json.loads(mp.read_text()) if mp.exists() else []
    write_jsonl(out / 'memory.jsonl', {'summary': True, 'benchmark': t, 'after_block': last, 'items': len(items),
                                       'active': sum(i['status'] == 'active' for i in items),
                                       'hold': sum(i['status'] == 'hold' for i in items), **{k: head[k] for k in (
                                           'ics_accuracy', 'vanilla_accuracy', 'difference_pp')}}, items)
    curve, usd = [], 0.0
    for k in range(1, last + 1):
        br = [r for r in rows if r['block'] == k]
        sp = c.root / 'stages' / t / f'block_{k:04d}.json'
        rec = json.loads(sp.read_text()) if sp.exists() else {}
        curve.append({'block': k, **part(br), 'example_runs': sum(bool(r['tried']) for r in br),
                      'memory_items': len(rec.get('memory_after', [])), 'teacher_usd_cumulative': rec.get('costs', {}).get('teacher_usd')})
    write_jsonl(out / 'block_curve.jsonl', {k: head[k] for k in ('summary', 'benchmark', 'status', 'ics_accuracy',
                                                                  'vanilla_accuracy', 'difference_pp')}, curve)
    atomic_json(out / 'summary.json', {**head, **detail}, immutable=False)
    (out / 'summary.md').write_text(report(head, detail))
    heads = []
    for b in c.tracks:
        p = Path(c.cfg['results_dir']) / b / 'model_output.jsonl'
        if p.exists():
            heads.append(json.loads(open(p).readline()))
    write_jsonl(Path(c.cfg['results_dir']) / 'overall.jsonl', overall(heads), heads)
    return head


def overall(heads):
    q = sum(h['questions'] for h in heads)
    return {'summary': True, 'benchmarks': [f'{h["benchmark"]}: {h["status"]}' for h in heads], 'questions': q,
            'ics_accuracy': round(sum(h['ics_accuracy'] * h['questions'] for h in heads) / q, 4) if q else None,
            'vanilla_accuracy': round(sum(h['vanilla_accuracy'] * h['questions'] for h in heads) / q, 4) if q else None,
            'per_benchmark_difference_pp': {h['benchmark']: h['difference_pp'] for h in heads},
            'teacher_usd_total': round(sum(h['teacher_usd'] or 0 for h in heads), 2)}


def report(h, d):
    fmt = lambda v: '-' if v is None else f'{v * 100:.2f}%' if isinstance(v, float) and v <= 1 else str(v)
    lines = [f'# {h["benchmark"]}: 확신도 게이트 + 진화 메모리 vs vanilla', '',
             f'- 상태: {h["status"]} · 작성 {h["made_at"]}',
             f'- 규칙: vanilla 답의 확신도(p1) < {h["rule"].split("p1 < ")[1].split(" ")[0]} 이거나 읽을 수 없으면 '
             f'유사도 1위 예시 하나를 주고 다시 풂. 다시 푼 답이 무효면 vanilla 답 유지.', '',
             '| | 문항 | ICS 정확도 | vanilla 정확도 | 차이 (%p) | 고침 | 망침 |', '|---|---|---|---|---|---|---|',
             f'| 전체 | {h["questions"]} | {fmt(h["ics_accuracy"])} | {fmt(h["vanilla_accuracy"])} | '
             f'{h["difference_pp"]} | {h["fixes"]} | {h["breaks"]} |']
    for name, key in (('예시를 받음', 'with_example'), ('저확신이나 쓸 예시 없음', 'gated_no_item'), ('고확신 (예시 없음)', 'not_gated')):
        r = d[key]
        if r['questions']:
            lines.append(f'| {name} | {r["questions"]} | {fmt(r["ics_accuracy"])} | {fmt(r["vanilla_accuracy"])} | '
                         f'{r["net_pp"]} | {r["fixes"]} | {r["breaks"]} |')
    lines += ['', f'- 차이의 95% 신뢰구간 (영상 단위 bootstrap): {h["ci95_pp"]} %p · 부호 검정 p = {h["sign_test_p"]:.3g}',
              f'- 예시를 준 문항 {h["example_runs"]} / 저확신 {h["gated"]} (쓸 예시가 없던 문항 {h["gated_without_item"]})',
              f'- 예시 풀이가 무효라 vanilla 답을 유지: {h["invalid_example_output_kept_vanilla"]} · 최종 무효 '
              f'{h["invalid_final"]} (vanilla 무효 {h["invalid_vanilla"]})',
              f'- 보여 준 예시의 정답 문장이 현재 문제의 정답 문장과 같은 경우: {h["shown_example_with_same_answer_text"]}',
              f'- teacher (sol) 비용: ${h["teacher_usd"]} · 호출 {h["teacher_calls"]}', '',
              '## 확신도 구간별', '', '| p1 구간 | 문항 | 예시 풀이 | ICS | vanilla | 차이 (%p) | 고침 | 망침 |',
              '|---|---|---|---|---|---|---|---|']
    for k, r in d['p1_bands'].items():
        if r['questions']:
            lines.append(f'| {k} | {r["questions"]} | {r["example_runs"]} | {fmt(r["ics_accuracy"])} | '
                         f'{fmt(r["vanilla_accuracy"])} | {r["net_pp"]} | {r["fixes"]} | {r["breaks"]} |')
    lines += ['', '고침/망침과 메모리 사용 통계는 연관일 뿐 인과가 아님. 고확신 문항은 vanilla 입력과 바이트 단위로 같아 차이가 0.']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/final_c.yaml')
    a = ap.parse_args()
    import run_icl_confidence as RC
    c = RC.setup(gpu=False, config=a.config)
    for t in c.tracks:
        rows, last = load_rows(c, t)
        h = export(c, t, finished=last == len(c.blocks[t]))
        print(t, None if h is None else {k: h[k] for k in ('status', 'ics_accuracy', 'vanilla_accuracy', 'difference_pp')})


if __name__ == '__main__':
    main()
