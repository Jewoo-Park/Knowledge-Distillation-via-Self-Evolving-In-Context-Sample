"""Convert dataset/VideoSIS/eval.jsonl into the framework's sample format (CPU only).

Each row becomes a Sample: absolute ordered frame_paths (the 32 provided frames), question, choices, ground_truth;
task_type, question_id and video_id go to metadata (never shown to the student); sample ids are neutral
(videosis:qNNNNN in source order) because the source question_id names the task type. The stream order is a fixed shuffle
(seed 20261026) because the source file is grouped by task type, which would make 50-question blocks single-type.

Writes data/videosis/stream.jsonl (checksummed lines, loadable with VideoQADataset(..., 'test')) and
data/videosis/summary.json.

    python3 scripts/prepare_videosis.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.storage import append_jsonl, atomic_json  # noqa: E402

SOURCE = ROOT / 'dataset' / 'VideoSIS' / 'eval.jsonl'
OUT = ROOT / 'data' / 'videosis'
SEED = 20261026


def main():
    rows = [json.loads(line) for line in open(SOURCE)]
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate question_id in the source')
    samples, problems = [], Counter()
    for n, r in enumerate(rows, 1):
        frames = [str((SOURCE.parent / f).resolve()) for f in r['frames']]
        missing = [f for f in frames if not Path(f).is_file()]
        if missing:
            problems['missing frames'] += 1
            continue
        if len(set(frames)) != len(frames) or r['answer'] not in r['options']:
            problems['bad frames or answer'] += 1
            continue
        # neutral id: the source question_id names the task type
        samples.append({'sample_id': f'videosis:q{n:05d}', 'dataset_name': 'VideoSIS', 'split': 'test',
                        'video_path': None, 'frame_paths': frames, 'question': r['question'].strip(),
                        'choices': {k: v.strip() for k, v in r['options'].items()}, 'ground_truth': r['answer'],
                        'metadata': {'benchmark_adapter': 'videosis', 'question_id': r['question_id'],
                                     'video_id': r['video_id'], 'task_type': r['task_type']}})
    random.Random(SEED).shuffle(samples)
    stream = OUT / 'stream.jsonl'
    if stream.exists():
        raise SystemExit(f'{stream} exists; it is the fixed stream of later runs, refusing to rewrite it')
    for s in samples:
        append_jsonl(stream, s)
    summary = {'source': str(SOURCE.relative_to(ROOT)), 'questions_in_source': len(rows), 'samples': len(samples),
               'skipped': dict(problems), 'videos': len({s['metadata']['video_id'] for s in samples}),
               'task_types': dict(Counter(s['metadata']['task_type'] for s in samples).most_common()),
               'answers': dict(Counter(s['ground_truth'] for s in samples)), 'shuffle_seed': SEED,
               'first_block_task_types': dict(Counter(s['metadata']['task_type'] for s in samples[:50]))}
    atomic_json(OUT / 'summary.json', summary, immutable=False)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
