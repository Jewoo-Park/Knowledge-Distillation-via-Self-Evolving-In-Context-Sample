"""Five-stage ICS checks: stage-2 isolation, option relabelling, retrieval follows when_to_use, memory operations."""
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.icl.prompts import permute_options
from src.icl.stage_memory import StageMemory
from src.icl.stages import AddOp, HoldOp, MergeOp, PruneOp, RefineOp, Stage5Output, teacher_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
CHOICES = {'A': 'left of the pond', 'B': 'above the pond', 'C': 'far from the pond'}
GOOD = '<THINK>The pond stays on the right in the last frames, so option A fits, not option (B).</THINK>\n<ANSWER>A</ANSWER>'


def cand(cid='cand-001-1', kind='corrected'):
    return {'candidate_id': cid, 'kind': kind, 'sample_id': 's1', 'video': 'v1', 'question': 'Where are you?',
            'choices': CHOICES, 'ground_truth': 'A', 'source_sample_ids': ['s1'], 'worked_output': GOOD,
            'supported_facts': ['f'], 'transferable_operation': 'op', 'risks': 'r', 'when_to_use': 'w1', 'strategy': 's',
            'trial_summary': [], 'next_action': 'memory_review'}


def s5(**kw):
    base = dict(block_analysis='', add=[], refine=[], merge=[], prune=[], hold=[])
    base.update(kw)
    return Stage5Output(**base)


def test_permutation_keeps_letters_consistent():
    for seed in range(30):
        choices, text = permute_options(CHOICES, GOOD, random.Random(seed))
        answer = re.search(r'<ANSWER>(.)</ANSWER>', text).group(1)
        assert choices[answer] == CHOICES['A']
        first = re.search(r'option ([A-E])', text).group(1)
        assert first == answer
        paren = re.search(r'\(([A-E])\)', text).group(1)
        assert choices[paren] == CHOICES['B']


def test_teacher_view_has_only_frames_question_options():
    x = SimpleNamespace(frame_paths=['a.jpg', 'b.jpg'], question='Q', choices=CHOICES, ground_truth='A',
                        metadata={'secret': 1}, sample_id='s1')
    assert set(teacher_view(x)) == {'frame_paths', 'question', 'choices'}
    import run_icl_stages as S
    with pytest.raises(RuntimeError):
        S.stage2(SimpleNamespace(cfg={'student': {'video': {}}}), [dict(teacher_view(x), ground_truth='A')])


def test_memory_ops_versions_and_validation():
    m = StageMemory('urban', max_active=2)
    cands = {'cand-001-1': cand(), 'cand-001-2': cand('cand-001-2', 'success'), 'cand-001-3': cand('cand-001-3')}
    applied, rejected = m.apply(s5(add=[AddOp(candidate_id=k, status='active', reason='r') for k in cands]), cands, 1)
    assert [a['item_id'] for a in applied] == ['ics-0001', 'ics-0002'] and 'active items' in rejected[0]['rejected_because']
    bad = '<THINK>x</THINK>\n<ANSWER>B</ANSWER>'
    applied, rejected = m.apply(s5(refine=[RefineOp(item_id='ics-0001', when_to_use='w2', strategy='', worked_output='', reason='r'),
                                           RefineOp(item_id='ics-0002', when_to_use='', strategy='', worked_output=bad, reason='r')]), {}, 2)
    assert m.items['ics-0001']['version'] == 2 and m.items['ics-0001']['history'][-1]['before'] == {'when_to_use': 'w1'}
    assert len(rejected) == 1 and m.items['ics-0002']['version'] == 1
    m.record_usage([{'shown': [{'id': 'ics-0001', 'version': 2}], 'correct': False, 'vanilla_correct': True}])
    assert m.items['ics-0001']['stats'] == {'2': {'shown': 1, 'correct': 0, 'gains': 0, 'harms': 1}}
    m.apply(s5(hold=[HoldOp(item_id='ics-0002', reason='thin')]), {}, 3)
    assert [i['id'] for i in m.active()] == ['ics-0001']
    applied, _ = m.apply(s5(add=[AddOp(candidate_id='ics-0002', status='active', reason='back')]), {}, 4)
    assert applied[0]['op'] == 'reactivate' and m.items['ics-0002']['status'] == 'active'
    applied, rejected = m.apply(s5(merge=[MergeOp(keep_id='ics-0001', drop_ids=['ics-0002'], when_to_use='w3', worked_output='',
                                                  reason='dup')], prune=[]), {}, 5)
    assert m.items['ics-0002']['status'] == 'merged' and m.items['ics-0001']['when_to_use'] == 'w3'
    m.apply(s5(prune=[PruneOp(item_id='ics-0001', reason='bad')]), {}, 6)
    assert not m.active() and len(m.items) == 2


def test_retrieval_follows_refined_when_to_use():
    import run_icl_stages as S
    vecs = {'w-near': np.array([1.0, 0.0]), 'w-far': np.array([0.0, 1.0])}
    x = SimpleNamespace(sample_id='q', question='new', choices={'A': 'x', 'B': 'y'})
    c = SimpleNamespace(vec={'q': np.array([1.0, 0.0])}, video={'q': 'vq'}, rule={'urban': {'threshold': 0.5, 'margin': 0.0}},
                        cfg={'retrieval': {'exclude_option_text': True}}, enc_lock=__import__('threading').Lock(),
                        retriever=SimpleNamespace(text=SimpleNamespace(encode=lambda t: vecs[t])))
    item = {'id': 'ics-0001', 'version': 1, 'video': 'v1', 'question': 'old', 'choices': CHOICES, 'ground_truth': 'A',
            'when_to_use': 'w-far'}
    assert S.retrieve(c, 'urban', x, [item])[0] == []
    item.update(when_to_use='w-near', version=2)
    got, ranking = S.retrieve(c, 'urban', x, [item])
    assert got == [item] and ranking[0]['version'] == 2
    same_text = dict(item, choices={'A': 'x', 'B': 'z'})  # its correct option text is one of the new options
    assert S.retrieve(c, 'urban', x, [same_text])[0] == []
    # coverage mode: the two best matches once the top-1 clears the quantile of past top-1 scores
    other = dict(item, id='ics-0002', when_to_use='w-far')
    c.rule['urban'] = {'mode': 'coverage', 'coverage': 0.75, 'threshold': None}
    c.cfg['retrieval'].update(max_examples=2, min_history=2)
    assert S.retrieve(c, 'urban', x, [item, other])[0] == []  # no history yet: nothing shown
    c.past_top = {'urban': [0.2, 0.5, 0.6, 0.7]}
    S.update_rule(c, 'urban')
    assert [i['id'] for i in S.retrieve(c, 'urban', x, [item, other])[0]] == ['ics-0001', 'ics-0002']



def test_retrieval_by_source_question_with_group_exclusion():
    import run_icl_stages as S
    x = SimpleNamespace(sample_id='q', question='new', choices={'A': 'x', 'B': 'y'})
    c = SimpleNamespace(vec={'q': np.array([1.0, 0.0]), 's1': np.array([0.8, 0.6]), 's2': np.array([0.6, 0.8]),
                             's3': np.array([1.0, 0.0])},
                        video={'q': '/f/EmbodiedCity_7_2'}, rule={'urban': {'mode': 'absolute', 'threshold': 0.7}},
                        cfg={'retrieval': {'exclude_option_text': True, 'key': 'question', 'exclude': 'group'}})
    near = {'id': 'ics-0001', 'version': 1, 'video': '/f/RealWorld_3', 'sample_id': 's1', 'question': 'a', 'choices': CHOICES,
            'ground_truth': 'A', 'when_to_use': 'unused'}
    far = dict(near, id='ics-0002', sample_id='s2', question='b')
    same_video = dict(near, id='ics-0003', sample_id='s3', question='c', video='/f/EmbodiedCity_7_0')  # another clip, same video
    got, ranking = S.retrieve(c, 'urban', x, [far, near, same_video])
    assert got == [near] and [r['id'] for r in ranking] == ['ics-0001', 'ics-0002']  # ranked by source-question cosine
    c.rule['urban']['threshold'] = 0.9
    assert S.retrieve(c, 'urban', x, [far, near])[0] == []  # top-1 0.8 below the absolute threshold


def test_stage5_can_refine_a_candidate_added_in_the_same_answer():
    m = StageMemory('urban')
    cands = {'cand-001-1': cand()}
    applied, rejected = m.apply(s5(add=[AddOp(candidate_id='cand-001-1', status='hold', reason='r')],
                                   refine=[RefineOp(item_id='cand-001-1', when_to_use='w9', strategy='', worked_output='', reason='r')]),
                                cands, 1)
    assert not rejected and m.items['ics-0001']['when_to_use'] == 'w9' and m.items['ics-0001']['version'] == 2


def test_one_sample_text_is_the_same_alone_or_in_a_list():
    import run_icl_stages as S
    c = SimpleNamespace(cfg={'retrieval': {'permute_options': True, 'position': 'after_video'}})
    x = SimpleNamespace(sample_id='q', question='Where?', choices={'A': 'x', 'B': 'y'})
    item = {'id': 'ics-0001', 'question': 'Old?', 'choices': CHOICES, 'worked_output': GOOD}
    assert S.parts_for(c, x, item) == S.parts_for(c, x, [item])
    two = S.parts_for(c, x, [item, dict(item, id='ics-0002')])
    assert two[0] == S.VIDEO and two[1].count('Example ') == 2
