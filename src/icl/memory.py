"""One benchmark's in-context sample memory: success and corrected items, edited only by validated critic ops.

Items are never deleted: prune and merge mark them inactive and keep their history. A snapshot of the whole memory
is written after every block (the memory used by block k+1 is the snapshot of block k).
"""
import copy
from src.evaluation.answer_parser import parse_answer
from src.icl.prompts import clean_output

KINDS = ('success', 'corrected')
PREFIX = {'success': 'succ', 'corrected': 'corr'}


def _stats():
    return {'shown': 0, 'correct_when_shown': 0, 'gains': 0, 'harms': 0}


class Memory:
    def __init__(self, benchmark, items=None):
        self.benchmark = benchmark
        self.items = {i['id']: i for i in (items or [])}

    def snapshot(self):
        return [copy.deepcopy(self.items[k]) for k in sorted(self.items)]

    def active(self, kind=None):
        return [i for k, i in sorted(self.items.items()) if i['status'] == 'active' and (kind is None or i['kind'] == kind)]

    def next_id(self, kind):
        n = sum(1 for i in self.items.values() if i['kind'] == kind) + 1
        return f'{PREFIX[kind]}-{n:04d}'

    def record_usage(self, rows):
        """rows: this block's results with shown item ids, student and vanilla correctness."""
        for r in rows:
            for item_id in r['shown_ids']:
                s = self.items[item_id]['stats']
                s['shown'] += 1
                s['correct_when_shown'] += int(r['correct'])
                s['gains'] += int(r['correct'] and not r['vanilla_correct'])
                s['harms'] += int(r['vanilla_correct'] and not r['correct'])

    # ---------------------------------------------------------------- validated operations
    def apply(self, out, cases, block, max_add):
        """out: CriticOutput; cases: {case_id: case}. -> (applied, rejected) as lists of dicts."""
        applied, rejected = [], []
        used_cases = {i['sample_id'] for i in self.items.values()}
        added = {k: 0 for k in KINDS}

        def reject(op, kind, why):
            rejected.append({'op': kind, **op.model_dump(), 'rejected_because': why})

        for op in out.add:
            case = cases.get(op.case_id)
            why = self._check_add(op, case, used_cases, added, max_add)
            if why:
                reject(op, 'add', why)
                continue
            example = clean_output(case['output']) if op.kind == 'success' else clean_output(op.corrected_output)
            item_id = self.next_id(op.kind)
            self.items[item_id] = {
                'id': item_id, 'kind': op.kind, 'status': 'active', 'version': 1, 'benchmark': self.benchmark,
                'sample_id': case['sample_id'], 'video': case['video'], 'question': case['question'],
                'choices': case['choices'], 'ground_truth': case['ground_truth'], 'when_to_use': op.when_to_use.strip(),
                'strategy': op.strategy.strip(), 'example': example,
                'source': {'student_output': case['output'], 'student_correct': case['correct'],
                           'teacher_answer': case['teacher_answer'], 'teacher_confidence': case['teacher_confidence']},
                'created_block': block, 'updated_block': block, 'merged_into': None, 'stats': _stats(),
                'history': [{'block': block, 'op': 'add', 'reason': op.reason}]}
            used_cases.add(case['sample_id'])
            added[op.kind] += 1
            applied.append({'op': 'add', 'item_id': item_id, 'kind': op.kind, 'case_id': op.case_id, 'reason': op.reason})
        for op in out.refine:
            item = self.items.get(op.item_id)
            if not item or item['status'] != 'active':
                reject(op, 'refine', 'unknown or inactive item')
                continue
            change = {}
            if op.when_to_use.strip():
                change['when_to_use'] = op.when_to_use.strip()
            if op.strategy.strip():
                change['strategy'] = op.strategy.strip()
            if op.corrected_output.strip():
                example = clean_output(op.corrected_output)
                if item['kind'] != 'corrected':
                    reject(op, 'refine', 'only corrected examples can be rewritten')
                    continue
                if not example or parse_answer(example, item['choices']).answer != item['ground_truth']:
                    reject(op, 'refine', 'rewritten example does not answer the correct option')
                    continue
                change['example'] = example
            if not change:
                reject(op, 'refine', 'no change')
                continue
            item.update(change, version=item['version'] + 1, updated_block=block)
            item['history'].append({'block': block, 'op': 'refine', 'fields': sorted(change), 'reason': op.reason})
            applied.append({'op': 'refine', 'item_id': op.item_id, 'fields': sorted(change), 'reason': op.reason})
        for op in out.merge:
            keep = self.items.get(op.keep_id)
            drops = [self.items.get(d) for d in op.drop_ids]
            if (not keep or keep['status'] != 'active' or not drops or any(not d or d['status'] != 'active' for d in drops)
                    or any(d['kind'] != keep['kind'] for d in drops) or op.keep_id in op.drop_ids):
                reject(op, 'merge', 'items unknown, inactive, of different kinds or keep among drops')
                continue
            if op.when_to_use.strip():
                keep.update(when_to_use=op.when_to_use.strip(), version=keep['version'] + 1)
            keep['updated_block'] = block
            keep['history'].append({'block': block, 'op': 'merge', 'absorbed': list(op.drop_ids), 'reason': op.reason})
            for d in drops:
                d.update(status='merged', merged_into=op.keep_id, updated_block=block)
                d['history'].append({'block': block, 'op': 'merged_into', 'target': op.keep_id, 'reason': op.reason})
            applied.append({'op': 'merge', 'keep_id': op.keep_id, 'drop_ids': list(op.drop_ids), 'reason': op.reason})
        for op in out.prune:
            item = self.items.get(op.item_id)
            if not item or item['status'] != 'active':
                reject(op, 'prune', 'unknown or inactive item')
                continue
            item.update(status='pruned', updated_block=block)
            item['history'].append({'block': block, 'op': 'prune', 'reason': op.reason})
            applied.append({'op': 'prune', 'item_id': op.item_id, 'reason': op.reason})
        return applied, rejected

    def _check_add(self, op, case, used_cases, added, max_add):
        if case is None:
            return 'unknown case_id'
        if case['sample_id'] in used_cases:
            return 'case already in memory'
        if added[op.kind] >= max_add:
            return f'more than {max_add} {op.kind} adds in one block'
        if not op.when_to_use.strip():
            return 'empty when_to_use'
        if not case['teacher_correct']:
            return 'teacher was not right'
        if op.kind == 'success':
            if not case['correct']:
                return 'student was not right'
            if not clean_output(case['output']):
                return 'student output lacks THINK/ANSWER blocks'
            return None
        if case['correct'] or not case['valid']:
            return 'corrected needs a valid wrong student answer'
        if case['teacher_confidence'] == 'low':
            return 'teacher confidence low'
        example = clean_output(op.corrected_output)
        if not example or parse_answer(example, case['choices']).answer != case['ground_truth']:
            return 'corrected example does not answer the correct option'
        student = clean_output(case['output']) or case['output']
        if len(example) > 2 * len(student) + 300:
            return 'corrected example far longer than the student output'
        return None
