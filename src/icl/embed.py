"""WeMM-Embedding-2B query vectors (video + question + options) for a new benchmark stream.

Same encoder, preprocessing and cache keys as the vectors precomputed for Urban/TempCompass/Video-MME
(qwen_vl_utils with an ordered frame list, Sentence-Transformers chat template, <embedding> pooling), restored from
archive/code_before_icl_evolution_20261026.tar.gz (src/evolving/retrieval.py). Used by scripts/embed_queries.py.
"""
import time
from pathlib import Path
import numpy as np
from src.icl.retrieval import normalize
from src.utils.storage import digest, file_hash


def query_text(instruction, question, choices):
    text = f'{instruction}\nQuestion: {question}'
    if choices:
        text += '\nOptions:\n' + '\n'.join(f'{k}. {v}' for k, v in choices.items())
    return text


def select_frames(paths, num_frames):
    """Same uniform indices as src.datasets.video_loader.load_frames."""
    indices = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return [paths[i] for i in indices], indices.tolist()


def resolve_dtype(config, device):
    dtype = config['dtype']
    if dtype == 'auto':
        import torch
        dtype = 'bfloat16' if device.startswith('cuda') and torch.cuda.get_device_capability(device)[0] >= 8 else 'float16'
    if device == 'cpu' and dtype == 'float16':
        dtype = 'float32'
    return dtype


_HASHES = {}


def _file_hash(path):
    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = file_hash(path)
    return _HASHES[key]


def wemm_signature(config, video, dtype):
    """Encoder identity without loading weights; part of every cache key."""
    import transformers
    path = Path(config['local_path'])
    return {'backend': 'wemm', 'model': config['model'], 'revision': config.get('revision'),
            'remote_code_sha256': _file_hash(path / 'modeling_wemm_embedding.py'),
            'weights_sha256': _file_hash(path / 'model.safetensors'), 'tokenizer_sha256': _file_hash(path / 'tokenizer.json'),
            'template_sha256': digest((path / config['chat_template']).read_text()), 'dtype': dtype,
            'dimension': config['dimension'], 'transformers_version': transformers.__version__,
            'min_pixels': config['min_pixels'], 'num_frames': video['num_frames'], 'max_pixels': video['max_pixels'],
            'assumed_sampled_fps': video['assumed_sampled_fps']}


class WeMMEncoder:
    def __init__(self, config, video, device, signature=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.config, self.video, self.device = config, video, device
        path = Path(config['local_path'])
        dtype = resolve_dtype(config, device)
        self.dtype = getattr(torch, dtype)
        self.template = (path / config['chat_template']).read_text()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        # trust_remote_code loads modeling_wemm_embedding.py from the pinned local snapshot;
        # its hash is part of the encoder signature.
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                               dtype=self.dtype, attn_implementation='eager').to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.patch_size = int(self.model.config.vision_config.patch_size)
        self.load_peak_bytes = torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0
        self.signature = signature or wemm_signature(config, video, dtype)
        self.last_stats = {}

    def _embed(self, content):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                  chat_template=self.template)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=self.patch_size,
                                                           return_video_kwargs=True, return_video_metadata=True)
        metadata = None
        if videos is not None:
            videos, metadata = (list(part) for part in zip(*videos))
        inputs = self.processor(text=[text], images=images, videos=videos, video_metadata=metadata,
                                return_tensors='pt', **video_kwargs)
        ids = inputs['input_ids'][0]
        if self.processor.tokenizer.convert_ids_to_tokens(int(ids[-1])) != '<embedding>':
            raise RuntimeError('WeMM input does not end with <embedding>')
        inputs = inputs.to(self.device)
        if 'pixel_values_videos' in inputs:
            inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(self.dtype)
        cuda = self.device.startswith('cuda')
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            vector = self.model.embedding(**inputs)[0].float().cpu().numpy()
        if not np.isfinite(vector).all():
            raise RuntimeError('WeMM produced a non-finite embedding (check EMBEDDING_DTYPE)')
        self.last_stats = {'input_tokens': int(ids.shape[0]), 'latency_seconds': time.perf_counter() - start,
            'video_grid_thw': inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else None,
            'video_tokens': int((ids == self.processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum()),
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if cuda else 0}
        return normalize(vector[:self.config['dimension']])

    def encode_text(self, text):
        return self._embed([{'type': 'text', 'text': text}])

    def encode_query(self, model_input, mode='video_text'):
        """mode: video_text (default), text (question+options only) or video (frames + instruction only)."""
        if mode == 'text':
            return self._embed([{'type': 'text', 'text': query_text(self.config['query_instruction'],
                                model_input['question'], model_input.get('choices'))}])
        v = self.video
        if model_input.get('frame_paths'):
            frames, _ = select_frames(model_input['frame_paths'], v['num_frames'])
            video = {'type': 'video', 'video': frames, 'sample_fps': v['assumed_sampled_fps']}
        else:
            video = {'type': 'video', 'video': model_input['video_path'], 'nframes': v['num_frames']}
        video.update(min_pixels=self.config['min_pixels'], max_pixels=v['max_pixels'])
        if mode == 'video':
            return self._embed([video, {'type': 'text', 'text': self.config['query_instruction']}])
        text = query_text(self.config['query_instruction'], model_input['question'], model_input.get('choices'))
        return self._embed([video, {'type': 'text', 'text': text}])

    def encode_frames(self, model_input, instruction):
        """Cosine of each sampled frame (image, document side) with the question+options query (visual_zoom)."""
        query = self._embed([{'type': 'text', 'text': query_text(instruction, model_input['question'],
                                                                  model_input.get('choices'))}])
        paths, _ = select_frames(model_input['frame_paths'], self.video['num_frames'])
        frames = np.stack([self._embed([{'type': 'image', 'image': p, 'min_pixels': self.config['min_pixels'],
                                         'max_pixels': self.video['max_pixels']}]) for p in paths])
        return (frames @ query).astype(np.float32)

    def close(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
