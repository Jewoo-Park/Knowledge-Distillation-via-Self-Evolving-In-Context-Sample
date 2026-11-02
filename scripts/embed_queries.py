"""WeMM query vectors (video + question + options) for every question of a config's stream, on one GPU per shard.

Writes cfg['query_cache'] (keys as src.icl.retrieval.query_key) and cfg['query_encoder_signature']. Resumable:
cached keys are skipped.

    source scripts/icl_env.sh && fuzzy35/bin/python scripts/embed_queries.py --config configs/final_c.yaml --gpu 2 --shard 0 --shards 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import yaml
    from src.datasets.base_dataset import VideoQADataset
    from src.icl.embed import WeMMEncoder
    from src.icl.retrieval import EmbeddingCache, query_key
    cfg = yaml.safe_load(Path(a.config).read_text())
    samples = sorted(VideoQADataset(cfg['stream'], 'test').samples, key=lambda s: s.sample_id)
    mine = [s for i, s in enumerate(samples) if i % a.shards == a.shard]
    encoder = WeMMEncoder(cfg['embedding'], cfg['student']['video'], 'cuda:0')
    sig_path = Path(cfg['query_encoder_signature'])
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    if sig_path.exists() and json.loads(sig_path.read_text()) != encoder.signature:
        raise SystemExit(f'encoder signature differs from {sig_path}')
    sig_path.write_text(json.dumps(encoder.signature, sort_keys=True))
    cache = EmbeddingCache(cfg['query_cache'])
    media, done, made, t0 = {}, 0, 0, time.time()
    for s in mine:
        key = query_key(encoder.signature, s, cfg['embedding']['query_instruction'], media)
        if cache.get(key) is None:
            cache.put(key, 'query', encoder.encode_query(s.model_input()))
            made += 1
        done += 1
        if done % 200 == 0 or done == len(mine):
            el = time.time() - t0
            print(f'shard {a.shard}: {done}/{len(mine)} (new {made}) {el / 60:.1f} min, eta {el / done * (len(mine) - done) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
