"""Paired comparison of two conditions on the same questions."""
import math
from collections import defaultdict
import numpy as np


def sign_test_p(gains, harms):
    """One-sided exact binomial P(X >= gains | n = gains + harms, p = 1/2)."""
    n = gains + harms
    return 1.0 if n == 0 else sum(math.comb(n, k) for k in range(gains, n + 1)) / 2 ** n


def paired_stats(base, treat, videos, n_boot=5000, seed=0):
    """Accuracy difference with a video-cluster bootstrap 95% CI, gains/harms and the exact sign test."""
    base, treat = np.asarray(base, float), np.asarray(treat, float)
    if not len(base):
        return {'n': 0, 'base_acc': None, 'acc': None, 'delta_pp': 0.0, 'ci95_pp': [0.0, 0.0], 'gains': 0, 'harms': 0,
                'sign_test_p': 1.0}
    d = treat - base
    groups = defaultdict(list)
    for k, v in enumerate(videos):
        groups[v].append(k)
    sums = np.array([d[g].sum() for g in groups.values()])
    sizes = np.array([len(g) for g in groups.values()])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sums), len(sums))
        boots.append(sums[pick].sum() / sizes[pick].sum())
    gains, harms = int(((treat == 1) & (base == 0)).sum()), int(((treat == 0) & (base == 1)).sum())
    return {'n': int(len(d)), 'base_acc': float(base.mean()), 'acc': float(treat.mean()), 'delta_pp': float(100 * d.mean()),
            'ci95_pp': [float(100 * np.percentile(boots, 2.5)), float(100 * np.percentile(boots, 97.5))],
            'gains': gains, 'harms': harms, 'sign_test_p': sign_test_p(gains, harms)}
