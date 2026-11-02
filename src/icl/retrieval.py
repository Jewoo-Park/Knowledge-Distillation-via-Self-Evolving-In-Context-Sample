"""Cosine retrieval of in-context samples with WeMM-Embedding-2B vectors.

Every question already has a cached WeMM query vector (video + question + options). key='question' (default)
scores an item by the cosine between the new question's vector and the vector of the item's source question;
key='when_to_use' scores it against the item's when_to_use text (WeMM text embedding, CPU, cached). Items need
score >= threshold; at most max_success success and max_corrected corrected items are returned, best first.
An item is never shown for a question on its own video or with the identical question and options.
"""
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
from src.datasets.base_dataset import media_hash
from src.utils.storage import digest, file_hash, now


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def query_key(encoder_signature, sample, query_instruction, media_cache=None):
    """Same key as the precomputed query vectors (encoder, media, question, options, instruction)."""
    model_input = sample.model_input()
    return digest({'encoder': encoder_signature, 'kind': 'query', 'media_sha256': media_hash(sample, media_cache),
                   'question': model_input['question'], 'choices': model_input.get('choices'),
                   'instruction': query_instruction})


class EmbeddingCache:
    def __init__(self, path, readonly=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        uri = f'file:{path}?mode=ro' if readonly else f'file:{path}'
        self.db = sqlite3.connect(uri, uri=True, timeout=60, isolation_level=None, check_same_thread=False)
        if not readonly:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, kind TEXT NOT NULL, '
                            'dim INTEGER NOT NULL, vector BLOB NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL)')

    def get(self, key):
        row = self.db.execute('SELECT dim, vector, sha256 FROM vectors WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise RuntimeError(f'Corrupt embedding cache entry {key}')
        return np.frombuffer(row[1], dtype=np.float32).reshape(row[0]).copy()

    def put(self, key, kind, vector):
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self.db.execute('INSERT OR IGNORE INTO vectors VALUES (?,?,?,?,?,?)',
                        (key, kind, len(vector), blob, hashlib.sha256(blob).hexdigest(), now()))


class TextEncoder:
    """WeMM text embedding for when_to_use keys, loaded on first use (CPU, float32)."""

    def __init__(self, config, cache):
        self.config, self.cache, self.model = config, cache, None
        path = Path(config['local_path'])
        self.signature = {'model': config['model'], 'revision': config.get('revision'), 'device': 'cpu',
                          'weights_sha256': file_hash(path / 'model.safetensors'),
                          'template_sha256': digest((path / config['chat_template']).read_text())}

    def encode(self, text):
        key = digest({'encoder': self.signature, 'kind': 'when_to_use', 'text': text})
        vector = self.cache.get(key)
        if vector is None:
            vector = self._embed(text)
            self.cache.put(key, 'when_to_use', vector)
        return normalize(vector)

    def _embed(self, text):
        import torch
        from transformers import AutoModel, AutoProcessor
        path = Path(self.config['local_path'])
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.template = (path / self.config['chat_template']).read_text()
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                                   dtype=torch.float32, attn_implementation='eager').eval()
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                    chat_template=self.template)
        inputs = self.processor(text=[prompt], return_tensors='pt')
        if self.processor.tokenizer.convert_ids_to_tokens(int(inputs['input_ids'][0][-1])) != '<embedding>':
            raise RuntimeError('WeMM input does not end with <embedding>')
        with torch.inference_mode():
            vector = self.model.embedding(**inputs)[0].float().numpy()
        return normalize(vector[:self.config['dimension']])


class Retriever:
    def __init__(self, query_vectors, config, text_encoder=None):
        """query_vectors: {sample_id: unit vector}; config: key, threshold, max_success, max_corrected."""
        self.vec, self.cfg, self.text = query_vectors, config, text_encoder
        if config['key'] not in ('question', 'when_to_use'):
            raise ValueError(f'unknown retrieval key {config["key"]}')

    def item_vector(self, item):
        return self.vec[item['sample_id']] if self.cfg['key'] == 'question' else self.text.encode(item['when_to_use'])

    def retrieve(self, sample, video, items):
        """-> (selected [(item, score)], ranking of every candidate [(id, kind, score)])."""
        q = self.vec[sample.sample_id]
        same = (sample.question.strip(), tuple(sorted(sample.choices.items())))
        scored = []
        for item in items:
            if item['video'] == video or (item['question'].strip(), tuple(sorted(item['choices'].items()))) == same:
                continue
            scored.append((item, float(self.item_vector(item) @ q)))
        scored.sort(key=lambda t: (-t[1], t[0]['id']))
        limits = {'success': self.cfg['max_success'], 'corrected': self.cfg['max_corrected']}
        selected = []
        for kind in ('success', 'corrected'):
            selected += [t for t in scored if t[0]['kind'] == kind and t[1] >= self.cfg['threshold']][:limits[kind]]
        return selected, [(i['id'], i['kind'], round(s, 4)) for i, s in scored[:20]]
