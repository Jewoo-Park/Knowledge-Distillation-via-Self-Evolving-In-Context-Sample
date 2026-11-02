"""Unit tests for the ICS framework (no GPU, no network)."""
import random
from types import SimpleNamespace
import numpy as np
from src.icl.memory import Memory
from src.icl.prompts import VIDEO, clean_output, permute_options, student_parts, user_prompt
from src.icl.retrieval import Retriever
from src.icl.stats import paired_stats, sign_test_p
from src.icl.teacher import AddOp, CriticOutput, MergeOp, PruneOp, RefineOp

CHOICES = {'A': 'left of the pond', 'B': 'above the pond', 'C': 'far from the pond'}
OUT = '<THINK>The pond is on the right (option A).</THINK>\n<ANSWER>A</ANSWER>'


def sample(sid, question='Where are you?', choices=CHOICES, gt='A'):
    return SimpleNamespace(sample_id=sid, question=question, choices=choices, ground_truth=gt)


def item(iid, kind, sid, video, score_vec=None, question='Q?'):
    return {'id': iid, 'kind': kind, 'sample_id': sid, 'video': video, 'question': question, 'choices': CHOICES,
            'ground_truth': 'A', 'example': OUT, 'when_to_use': 'w', 'version': 1}


def test_permute_options_moves_answer_with_text():
    for seed in range(20):
        choices, example = permute_options(CHOICES, OUT, random.Random(seed))
        letter = example.split('<ANSWER>')[1].split('</ANSWER>')[0]
        assert choices[letter] == CHOICES['A'] and sorted(choices.values()) == sorted(CHOICES.values())
        assert f'option {letter}' in example


def test_student_parts_vanilla_without_items():
    x = sample('s1')
    assert student_parts(x, []) is None
    parts = student_parts(x, [item('succ-0001', 'success', 's2', 'v2')])
    assert parts[1] == VIDEO and parts[2] == user_prompt(x) and parts[0].startswith('Solved examples')


def test_retriever_threshold_caps_and_exclusions():
    rng = np.random.default_rng(0)
    q = rng.normal(size=8); q /= np.linalg.norm(q)
    vec = {'new': q}
    items = []
    for n in range(8):
        v = q + rng.normal(scale=0.2 + 0.3 * n, size=8); v /= np.linalg.norm(v)
        vec[f's{n}'] = v
        items.append(item(f'{"succ" if n % 2 else "corr"}-{n:04d}', 'success' if n % 2 else 'corrected', f's{n}', f'v{n}'))
    vec['same'] = q
    items.append(item('succ-same-video', 'success', 'same', 'vnew'))
    vec['dup'] = q
    items.append(item('succ-dup', 'success', 'dup', 'vdup', question='Where are you?'))
    r = Retriever(vec, {'key': 'question', 'threshold': 0.5, 'max_success': 2, 'max_corrected': 1})
    selected, ranking = r.retrieve(sample('new'), 'vnew', items)
    ids = [i['id'] for i, _ in selected]
    assert 'succ-same-video' not in ids and 'succ-dup' not in ids
    assert all(s >= 0.5 for _, s in selected)
    kinds = [i['kind'] for i, _ in selected]
    assert kinds.count('success') <= 2 and kinds.count('corrected') <= 1
    assert kinds == sorted(kinds, key=lambda k: k != 'success')


def case(correct, teacher_correct=True, valid=True, output=OUT, conf='high'):
    return {'sample_id': f'x{correct}{teacher_correct}{valid}', 'video': 'v', 'question': 'Q', 'choices': CHOICES,
            'ground_truth': 'A', 'output': output, 'answer': 'A' if correct else 'B', 'valid': valid, 'correct': correct,
            'vanilla_correct': False, 'shown_ids': [], 'teacher_answer': 'A' if teacher_correct else 'B',
            'teacher_correct': teacher_correct, 'teacher_confidence': conf}


def test_memory_ops_are_validated():
    m = Memory('urban')
    wrong = '<THINK>It is above.</THINK>\n<ANSWER>B</ANSWER>'
    cases = {'Q01': case(True), 'Q02': case(False, output=wrong), 'Q03': case(False, teacher_correct=False, output=wrong)}
    out = CriticOutput(block_analysis='', add=[
        AddOp(kind='success', case_id='Q01', when_to_use='w', strategy='s', corrected_output='', reason='r'),
        AddOp(kind='corrected', case_id='Q02', when_to_use='w', strategy='s', corrected_output=OUT, reason='r'),
        AddOp(kind='corrected', case_id='Q03', when_to_use='w', strategy='s', corrected_output=OUT, reason='r'),
        AddOp(kind='success', case_id='Q02', when_to_use='w', strategy='s', corrected_output='', reason='r')],
        refine=[], merge=[], prune=[])
    applied, rejected = m.apply(out, cases, 1, max_add=5)
    assert [a['item_id'] for a in applied] == ['succ-0001', 'corr-0001'] and len(rejected) == 2
    bad = CriticOutput(block_analysis='', add=[], refine=[
        RefineOp(item_id='corr-0001', when_to_use='', strategy='', corrected_output=wrong, reason='r')], merge=[],
        prune=[PruneOp(item_id='succ-0001', reason='r')])
    applied, rejected = m.apply(bad, {}, 2, max_add=5)
    assert len(rejected) == 1 and m.items['succ-0001']['status'] == 'pruned'
    assert m.items['corr-0001']['example'] == OUT
    m.record_usage([{'shown_ids': ['corr-0001'], 'correct': True, 'vanilla_correct': False}])
    assert m.items['corr-0001']['stats'] == {'shown': 1, 'correct_when_shown': 1, 'gains': 1, 'harms': 0}
    bad_merge = CriticOutput(block_analysis='', add=[], refine=[], prune=[],
                             merge=[MergeOp(keep_id='corr-0001', drop_ids=['succ-0001'], when_to_use='w', reason='r')])
    assert len(m.apply(bad_merge, {}, 3, max_add=5)[1]) == 1


def test_clean_output_and_stats():
    assert clean_output('noise <THINK>a</THINK> <ANSWER>B</ANSWER>') == '<THINK>a</THINK>\n<ANSWER>B</ANSWER>'
    assert clean_output('<ANSWER>B</ANSWER>') is None
    st = paired_stats([0, 1, 0, 1], [1, 1, 1, 0], ['a', 'a', 'b', 'c'])
    assert st['gains'] == 2 and st['harms'] == 1 and round(st['delta_pp'], 1) == 25.0
    assert sign_test_p(0, 0) == 1.0
