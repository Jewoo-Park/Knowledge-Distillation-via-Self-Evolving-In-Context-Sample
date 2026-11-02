"""Memory of the five-stage ICS evolution: validated stage-5 operations with before/after records.

An item's version increases with every change of what it holds; online usage is counted per version. Nothing is
deleted: prune and merge change the status. Only 'active' items can be shown to the student.
"""
import copy
from src.evaluation.answer_parser import parse_answer
from src.icl.prompts import clean_output

FIELDS = ('when_to_use', 'strategy', 'worked_output')


def answers(text, choices, truth):
    example = clean_output(text or '')
    return example if example and parse_answer(example, choices).answer == truth else None


class StageMemory:
    def __init__(self, benchmark, items=None, max_active=20):
        self.benchmark, self.max_active = benchmark, max_active
        self.items = {i['id']: i for i in (items or [])}

    def snapshot(self):
        return [copy.deepcopy(self.items[k]) for k in sorted(self.items)]

    def active(self):
        return [i for _, i in sorted(self.items.items()) if i['status'] == 'active']

    def listed(self):
        return [i for _, i in sorted(self.items.items()) if i['status'] in ('active', 'hold')]

    def record_usage(self, rows):
        for r in rows:
            for s in r['shown']:
                item = self.items[s['id']]
                st = item['stats'].setdefault(str(s['version']), {'shown': 0, 'correct': 0, 'gains': 0, 'harms': 0})
                st['shown'] += 1
                st['correct'] += int(r['correct'])
                st['gains'] += int(r['correct'] and not r['vanilla_correct'])
                st['harms'] += int(r['vanilla_correct'] and not r['correct'])
                item['stats_all']['shown'] += 1

    def _change(self, item, block, op, reason, **fields):
        before = {k: item[k] for k in fields}
        item.update(fields)
        item.update(version=item['version'] + 1, updated_block=block)
        item['history'].append({'block': block, 'op': op, 'reason': reason, 'version': item['version'],
                                'before': before, 'after': dict(fields)})

    def apply(self, out, candidates, block):
        """out: Stage5Output; candidates: {candidate_id: final candidate}. -> (applied, rejected)."""
        applied, rejected = [], []

        def reject(kind, op, why):
            rejected.append({'op': kind, **op.model_dump(), 'rejected_because': why})

        self._add(out.add, candidates, block, applied, reject)
        alias = {cid: cand['added_as'] for cid, cand in candidates.items() if cand.get('added_as')}
        for op in list(out.prune) + list(out.hold) + list(out.refine):
            op.item_id = alias.get(op.item_id, op.item_id)
        for op in out.merge:
            op.keep_id = alias.get(op.keep_id, op.keep_id)
            op.drop_ids = [alias.get(d, d) for d in op.drop_ids]

        for op in out.prune:
            item = self.items.get(op.item_id)
            if not item or item['status'] not in ('active', 'hold'):
                reject('prune', op, 'unknown or already removed item')
                continue
            self._status(item, 'pruned', block, 'prune', op.reason)
            applied.append({'op': 'prune', 'item_id': op.item_id, 'reason': op.reason})
        for op in out.hold:
            item = self.items.get(op.item_id)
            if not item or item['status'] != 'active':
                reject('hold', op, 'unknown or not active item')
                continue
            self._status(item, 'hold', block, 'hold', op.reason)
            applied.append({'op': 'hold', 'item_id': op.item_id, 'reason': op.reason})
        for op in out.merge:
            keep, drops = self.items.get(op.keep_id), [self.items.get(d) for d in op.drop_ids]
            if (not keep or keep['status'] not in ('active', 'hold') or not drops or op.keep_id in op.drop_ids
                    or any(not d or d['status'] not in ('active', 'hold') for d in drops)):
                reject('merge', op, 'unknown or removed items, or keep among drops')
                continue
            change = {}
            if op.when_to_use.strip():
                change['when_to_use'] = op.when_to_use.strip()
            if op.worked_output.strip():
                example = answers(op.worked_output, keep['choices'], keep['ground_truth'])
                if not example:
                    reject('merge', op, 'worked_output does not answer the kept item\'s correct option')
                    continue
                change['worked_output'] = example
            if change:
                self._change(keep, block, 'merge', op.reason, **change)
            keep['history'].append({'block': block, 'op': 'absorbed', 'items': list(op.drop_ids), 'reason': op.reason})
            for d in drops:
                d['merged_into'] = op.keep_id
                self._status(d, 'merged', block, 'merged_into', op.reason)
            applied.append({'op': 'merge', 'keep_id': op.keep_id, 'drop_ids': list(op.drop_ids), 'fields': sorted(change),
                            'reason': op.reason})
        for op in out.refine:
            item = self.items.get(op.item_id)
            if not item or item['status'] not in ('active', 'hold'):
                reject('refine', op, 'unknown or removed item')
                continue
            change = {k: getattr(op, k).strip() for k in FIELDS if getattr(op, k).strip() and getattr(op, k).strip() != item[k]}
            if 'worked_output' in change:
                example = answers(change['worked_output'], item['choices'], item['ground_truth'])
                if not example:
                    reject('refine', op, 'worked_output does not answer the correct option')
                    continue
                change['worked_output'] = example
            if not change:
                reject('refine', op, 'no change')
                continue
            self._change(item, block, 'refine', op.reason, **change)
            applied.append({'op': 'refine', 'item_id': op.item_id, 'fields': sorted(change), 'reason': op.reason})
        return applied, rejected

    def _add(self, ops, candidates, block, applied, reject):
        for op in ops:
            if op.candidate_id in self.items:  # re-activate a held item
                item = self.items[op.candidate_id]
                if item['status'] != 'hold' or op.status != 'active':
                    reject('add', op, 'only a held item can be re-activated')
                    continue
                if len(self.active()) >= self.max_active:
                    reject('add', op, f'already {self.max_active} active items')
                    continue
                self._status(item, 'active', block, 'reactivate', op.reason)
                applied.append({'op': 'reactivate', 'item_id': item['id'], 'reason': op.reason})
                continue
            cand = candidates.get(op.candidate_id)
            if not cand:
                reject('add', op, 'unknown candidate')
                continue
            if cand.get('added_as'):
                reject('add', op, 'candidate already added')
                continue
            if op.status == 'active' and len(self.active()) >= self.max_active:
                reject('add', op, f'already {self.max_active} active items')
                continue
            item_id = f'ics-{len(self.items) + 1:04d}'
            self.items[item_id] = {
                'id': item_id, 'kind': cand['kind'], 'status': op.status, 'version': 1, 'benchmark': self.benchmark,
                'sample_id': cand['sample_id'], 'video': cand['video'], 'question': cand['question'],
                'choices': cand['choices'], 'ground_truth': cand['ground_truth'], 'when_to_use': cand['when_to_use'],
                'strategy': cand['strategy'], 'worked_output': cand['worked_output'],
                'supported_facts': cand['supported_facts'], 'transferable_operation': cand['transferable_operation'],
                'risks': cand['risks'], 'candidate_id': op.candidate_id, 'source_case_ids': cand['source_sample_ids'],
                'created_block': block, 'updated_block': block, 'merged_into': None,
                'stats': {}, 'stats_all': {'shown': 0}, 'trials': cand['trial_summary'],
                'history': [{'block': block, 'op': 'add', 'status': op.status, 'reason': op.reason, 'version': 1,
                             'after': {k: cand[k] for k in FIELDS}}]}
            cand['added_as'] = item_id
            applied.append({'op': 'add', 'item_id': item_id, 'candidate_id': op.candidate_id, 'status': op.status,
                            'reason': op.reason})

    def _status(self, item, status, block, op, reason):
        item['history'].append({'block': block, 'op': op, 'reason': reason, 'status': {'from': item['status'], 'to': status}})
        item['status'], item['updated_block'] = status, block
