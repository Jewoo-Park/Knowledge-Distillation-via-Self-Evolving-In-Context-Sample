"""Local Qwen3.5 video inference with explicit NF4 and recorded thinking mode."""
import time
from pathlib import Path
from src.models.base_vlm import BaseVLM, Generation
from src.utils.reproducibility import seed_everything
from src.utils.storage import file_hash, digest


VIDEO_PART = '<VIDEO>'
ZOOM_TEXT = 'Close-ups of the moments most related to the question, at higher resolution:\n'


def zoom_time_label(index, fps):
    """Same label as the processor's timeline token for the temporal patch (two frames) holding this frame."""
    return f'{(2 * (index // 2) + 0.5) / fps:.1f} seconds'


def split_native_thinking(raw):
    """The opening native think tag is supplied by the chat template.

    Only a completed native block releases a final response for scoring. An
    ANSWER mentioned inside unfinished reasoning must never become a prediction.
    Raw generated text is retained unchanged by the caller.
    """
    reasoning, boundary, answer = raw.partition('</think>')
    if reasoning.startswith('<think>'):
        reasoning = reasoning[len('<think>'):]
    return reasoning.strip(), answer if boundary else '', bool(boundary)


class _LogprobRecorder:
    """Logits processor that only reads: per step, the log-softmax top-1 / top-2, the entropy and the option letters'
    log probabilities of the next-token distribution. Returns the scores unchanged (greedy picks the same token)."""

    def __init__(self, letter_ids):
        self.letter_ids, self.steps = letter_ids, []

    def __call__(self, input_ids, scores):
        import torch
        logp = torch.log_softmax(scores[0].float(), dim=-1)
        top = torch.topk(logp, 2)
        p = logp.exp()
        self.steps.append({'top1_id': int(top.indices[0]), 'top1': float(top.values[0]), 'top2': float(top.values[1]),
                           'entropy': float(-torch.where(p > 0, p * logp, torch.zeros_like(p)).sum()),
                           'letters': {a: float(logp[i]) for a, i in self.letter_ids.items()}})
        return scores


def checkpoint_fingerprint(path):
    path = Path(path).resolve()
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.safetensors'))
    if not files or not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('A complete local multimodal checkpoint is required')
    hashes = {p.name: file_hash(p) for p in files}
    return {'files': hashes, 'sha256': digest(hashes)}


class Qwen35VL(BaseVLM):
    def load_model(self):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
        device = self.config['device']
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Qwen3.5 experiments require a real CUDA GPU')
        if self.config['dtype'] != 'float16' or self.config['attention'] != 'eager':
            raise ValueError('These Turing experiments require float16 and eager attention')
        if self.config.get('cpu_offload'):
            raise ValueError('Matched comparison requires no CPU offload')
        self.thinking = self.config.get('enable_thinking', False)
        if not isinstance(self.thinking, bool):
            raise ValueError('enable_thinking must be an explicit boolean')
        quant = self.config.get('quantization')
        if quant not in (None, '4bit'):
            raise ValueError('Only full FP16 or explicit 4bit NF4 is supported')
        path = self.config['local_path']
        identity = checkpoint_fingerprint(path)
        if identity['sha256'] != self.config['checkpoint_sha256']:
            raise ValueError('Local checkpoint changed after configuration was frozen')
        kwargs = dict(dtype=torch.float16, attn_implementation='eager',
                      device_map={'': device}, local_files_only=True)
        if quant:
            import bitsandbytes as bnb
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
            self.runtime_identity['bitsandbytes_version'] = bnb.__version__
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        frames = self.video_config['num_frames']
        self.processor.video_processor.size = {'shortest_edge': frames * 4 * 32 * 32,
                                               'longest_edge': frames * self.video_config['max_pixels']}
        self.processor.video_processor.cap_pixels_per_frame = False
        self.runtime_identity.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
            checkpoint_sha256=identity['sha256'], thinking=self.thinking,
            processor_video_size=dict(self.processor.video_processor.size),
            quantization_parameters={'type': 'nf4', 'double_quant': True, 'compute_dtype': 'float16'} if quant else None)
        if quant:
            self.runtime_identity['linear4bit_modules'] = sum(isinstance(m, bnb.nn.Linear4bit) for m in self.model.modules())
            if not self.runtime_identity['linear4bit_modules'] or not getattr(self.model, 'is_loaded_in_4bit', False):
                raise RuntimeError('Requested 4bit quantization did not actually load')
        self.load_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)

    def zoom_content(self, model_input, metadata, zoom, prompt):
        """Close-up frames after the video (visual_zoom action): text, 'At t:' + image per frame, then the prompt."""
        import cv2
        from src.datasets.video_loader import _resize_rgb
        if not model_input.get('frame_paths'):
            raise NotImplementedError('visual_zoom needs frame_paths inputs')
        content, images = [{'type': 'video'}, {'type': 'text', 'text': ZOOM_TEXT}], []
        for i in zoom['frames']:
            path = model_input['frame_paths'][metadata['sampled_indices'][i]]
            content += [{'type': 'text', 'text': f'At {zoom_time_label(i, metadata["sampled_fps"])}:'}, {'type': 'image'},
                        {'type': 'text', 'text': '\n'}]
            images.append(_resize_rgb(cv2.imread(str(path)), zoom['pixels']))
        content.append({'type': 'text', 'text': '\n' + prompt})
        return content, images

    def generate_text(self, prompt, seed, system_prompt=None, max_new_tokens=256):
        """Text-only generation (answer_check post action); no video."""
        import torch
        seed_everything(seed)
        start = time.perf_counter()
        messages = self.chat_messages([{'type': 'text', 'text': prompt}], system_prompt)
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.processor(text=[chat], return_tensors='pt', padding=True).to(self.config['device'])
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return Generation(raw, {'latency_seconds': time.perf_counter() - start, 'input_tokens': inputs.input_ids.shape[1],
                                'generated_tokens': output.shape[1] - inputs.input_ids.shape[1]})

    def generate(self, model_input, prompt, seed, system_prompt=None, zoom=None, parts=None, turns=None,
                 logprobs=False, letters=()):
        """parts (optional): ordered user content, text strings and VIDEO_PART for the video; replaces prompt.
        The video tensor and its timeline are identical whatever the order.
        logprobs: also record, per generated token, its log probability, the entropy and the second-best log probability
        of the next-token distribution, and at the answer letter (the '>X' token after '<ANSWER') the log probabilities
        of the option `letters` (runtime['logprobs']). Only read from the logits; greedy decoding picks the same tokens."""
        import torch
        from transformers.video_utils import VideoMetadata
        seed_everything(seed)
        device = self.config['device']
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        frames, metadata = self.preprocess_media(model_input)
        images = []
        if parts is not None:
            if zoom or parts.count(VIDEO_PART) != 1:
                raise ValueError('parts need exactly one video part and no zoom')
            # {'frame': i, 'pixels': n}: close-up of sampled frame i (same source JPEG), resized to n pixels.
            import cv2
            from src.datasets.video_loader import _resize_rgb
            content = []
            for p in parts:
                if p == VIDEO_PART:
                    content.append({'type': 'video'})
                elif isinstance(p, dict):
                    path = model_input['frame_paths'][metadata['sampled_indices'][p['frame']]]
                    images.append(_resize_rgb(cv2.imread(str(path)), p['pixels']))
                    content.append({'type': 'image'})
                    self.processor.image_processor.size = {'shortest_edge': 4 * 32 * 32, 'longest_edge': p['pixels']}
                else:
                    content.append({'type': 'text', 'text': p})
        elif zoom:
            content, images = self.zoom_content(model_input, metadata, zoom, prompt)
            self.processor.image_processor.size = {'shortest_edge': 4 * 32 * 32, 'longest_edge': zoom['pixels']}
        else:
            content = [{'type': 'video'}, {'type': 'text', 'text': prompt}]
        def meta_for(clip, fps):
            return VideoMetadata(total_num_frames=len(clip), fps=fps, frames_indices=list(range(len(clip))),
                                 height=clip.shape[1], width=clip.shape[2])
        videos, metas = [frames], [meta_for(frames, metadata['sampled_fps'])]
        if turns is not None:
            # Few-shot conversation: earlier user/assistant turns (example videos as {'video_paths', 'num_frames'}),
            # the last user turn holds VIDEO_PART for this question. Videos are passed in order of appearance.
            from src.datasets.video_loader import load_frames
            if parts is not None or zoom or sum(VIDEO_PART in t['parts'] for t in turns) != 1:
                raise ValueError('turns need exactly one target video part and no parts/zoom')
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}] if system_prompt else []
            videos, metas = [], []
            for turn in turns:
                body = []
                for p in turn['parts']:
                    if p == VIDEO_PART:
                        body.append({'type': 'video'})
                        videos.append(frames)
                        metas.append(meta_for(frames, metadata['sampled_fps']))
                    elif isinstance(p, dict):
                        clip, info = load_frames(p['video_paths'], num_frames=p['num_frames'],
                                                 max_pixels=p.get('pixels', self.video_config['max_pixels']),
                                                 assumed_sampled_fps=self.video_config['assumed_sampled_fps'])
                        body.append({'type': 'video'})
                        videos.append(clip)
                        metas.append(meta_for(clip, info['sampled_fps']))
                    else:
                        body.append({'type': 'text', 'text': p})
                messages.append({'role': turn['role'], 'content': body})
        else:
            messages = self.chat_messages(content, system_prompt)
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking)
        expected_suffix = '<think>\n' if self.thinking else '<think>\n\n</think>\n\n'
        if not chat.endswith(expected_suffix):
            raise RuntimeError('Chat template did not honor the requested thinking mode')
        inputs = self.processor(text=[chat], videos=videos, video_metadata=metas,
            do_sample_frames=False, return_tensors='pt', padding=True, **({'images': images} if images else {}))
        if 'pixel_values_videos' not in inputs or 'video_grid_thw' not in inputs:
            raise RuntimeError('Qwen3.5 video tensors missing')
        inputs = inputs.to(device)
        # Pixel dtype must agree with the vision encoder on pre-Ampere hardware.
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        generation = dict(self.generation_config)
        if generation.pop('batch_size', 1) != 1:
            raise ValueError('Only batch size one is supported')
        if logprobs:
            if generation.get('do_sample'):
                raise ValueError('logprobs are recorded for greedy decoding only')
            from transformers import LogitsProcessorList
            recorder = _LogprobRecorder(self.letter_token_ids(letters))
            generation['logits_processor'] = LogitsProcessorList([recorder])
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        raw = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        token_logprobs = self.logprob_record(recorder, output[0, inputs.input_ids.shape[1]:].tolist()) if logprobs else None
        reasoning, final, closed = split_native_thinking(raw) if self.thinking else (None, None, None)
        generated_ids = output[0, inputs.input_ids.shape[1]:].tolist()
        think_end = self.processor.tokenizer.convert_tokens_to_ids('</think>')
        boundary_index = generated_ids.index(think_end) if self.thinking and think_end in generated_ids else None
        if self.thinking and closed != (boundary_index is not None):
            raise RuntimeError('Native thinking token boundary and decoded text disagree')
        torch.cuda.synchronize(device)
        return Generation(raw, {'mock': False, 'latency_seconds': time.perf_counter() - start,
            'video': metadata, 'chat_prompt': chat, 'thinking': self.thinking,
            'native_thinking_closed': closed,
            'native_thinking_tokens': boundary_index + 1 if boundary_index is not None else len(generated_ids) if self.thinking else 0,
            'final_response_tokens': len(generated_ids)-boundary_index-1 if boundary_index is not None else 0 if self.thinking else len(generated_ids),
            'hit_output_token_limit': len(generated_ids) >= generation.get('max_new_tokens', float('inf')),
            'processor_input_text': self.processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False),
            'input_tokens': inputs.input_ids.shape[1], 'generated_tokens': output.shape[1] - inputs.input_ids.shape[1],
            'load_peak_allocated_bytes': self.load_peak_allocated_bytes,
            'effective_generation_config': dict(self.model.generation_config.to_dict(), **generation),
            'video_tensor_shape': list(inputs['pixel_values_videos'].shape),
            'video_grid_thw': inputs['video_grid_thw'].tolist(), 'zoom': zoom,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            **({'logprobs': token_logprobs} if logprobs else {})},
            answer_output=final, native_reasoning=reasoning)

    def letter_token_ids(self, letters):
        """letter -> id of the single token '>' + letter ('>A' is one token in this tokenizer)."""
        ids = {}
        for a in letters:
            t = self.processor.tokenizer.encode('>' + a, add_special_tokens=False)
            if len(t) != 1:
                raise RuntimeError(f"'>{a}' is not a single token")
            ids[a] = t[0]
        return ids

    def logprob_record(self, recorder, generated_ids):
        """Per-token record of a logprobs generation, aligned with generated_ids, plus the answer letter's step."""
        import math
        tok = self.processor.tokenizer
        steps = recorder.steps
        if len(steps) != len(generated_ids):
            raise RuntimeError(f'{len(steps)} recorded steps for {len(generated_ids)} generated tokens')
        rows, mismatches = [], 0
        for i, s in zip(generated_ids, steps):
            greedy = s['top1_id'] == i  # always, for greedy decoding without score-changing processors
            mismatches += not greedy
            rows.append([tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                         round(s['top1'], 5) if greedy else None, round(s['entropy'], 5), round(s['top2'], 5)])
        answer = None
        by_id = {v: k for k, v in recorder.letter_ids.items()}
        for t, i in enumerate(generated_ids):
            if i in by_id and tok.decode(generated_ids[:t], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False).upper().endswith('<ANSWER'):
                lp = steps[t]['letters']
                z = max(lp.values())
                norm = z + math.log(sum(math.exp(v - z) for v in lp.values()))
                answer = {'step': t, 'letter': by_id[i], 'letter_logprobs': {a: round(v, 5) for a, v in lp.items()},
                          'probs': {a: round(math.exp(v - norm), 6) for a, v in lp.items()},
                          'letter_mass': round(sum(math.exp(v) for v in lp.values()), 6)}
                break
        return {'fields': ['token', 'logprob', 'entropy', 'top2_logprob'], 'tokens': rows, 'answer': answer,
                'not_greedy_steps': mismatches}

    def answer_scores(self, model_input, seed, system_prompt=None, parts=None, prefix='', letters=(), token_prefix=''):
        """Probabilities of the option letters as the next token after `prefix` (an earlier output of this model on the
        same input, cut where its answer letter starts), renormalised over `letters`; each candidate token is
        token_prefix + letter (this tokenizer writes '>A' as one token, so the cut is before the '>' of '<ANSWER>').
        One forward pass, no generation; the input is built exactly as in generate(parts=...)."""
        import json
        import torch
        from transformers.video_utils import VideoMetadata
        seed_everything(seed)
        device = self.config['device']
        start = time.perf_counter()
        frames, metadata = self.preprocess_media(model_input)
        if parts is None or parts.count(VIDEO_PART) != 1 or any(isinstance(p, dict) for p in parts):
            raise ValueError('answer_scores needs text parts with exactly one video part')
        content = [{'type': 'video'} if p == VIDEO_PART else {'type': 'text', 'text': p} for p in parts]
        chat = self.processor.apply_chat_template(self.chat_messages(content, system_prompt), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=self.thinking)
        if not chat.endswith('<think>\n' if self.thinking else '<think>\n\n</think>\n\n'):
            raise RuntimeError('Chat template did not honor the requested thinking mode')
        meta = VideoMetadata(total_num_frames=len(frames), fps=metadata['sampled_fps'], frames_indices=list(range(len(frames))),
                             height=frames.shape[1], width=frames.shape[2])
        inputs = self.processor(text=[chat + prefix], videos=[frames], video_metadata=[meta], do_sample_frames=False,
                                return_tensors='pt', padding=True).to(device)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
        ids = {}
        for a in letters:
            t = self.processor.tokenizer.encode(token_prefix + a, add_special_tokens=False)
            if len(t) != 1:
                raise RuntimeError(f'{token_prefix + a!r} is not a single token')
            ids[a] = t[0]
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        sub = torch.tensor([logp[ids[a]].item() for a in letters])
        probs = torch.softmax(sub, dim=-1).tolist()
        mass = float(torch.exp(sub).sum())  # share of the full next-token distribution on the option letters
        torch.cuda.synchronize(device)
        out = {'probs': dict(zip(letters, [round(p, 6) for p in probs])), 'letter_mass': round(mass, 6),
               'argmax': letters[int(torch.argmax(sub))]}
        return Generation(json.dumps(out), {'latency_seconds': time.perf_counter() - start,
                                            'input_tokens': inputs.input_ids.shape[1], 'generated_tokens': 0})

    def unload_model(self):
        import gc
        import torch
        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
