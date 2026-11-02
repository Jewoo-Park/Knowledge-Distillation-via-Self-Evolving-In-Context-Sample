"""One backbone per GPU in persistent worker processes, shared by a single coordinator.

The coordinator keeps retrieval, the example memory and the teacher; workers only generate. Results come back
in request order, so a block of questions answered in parallel is recorded exactly as a sequential
run would record it (same prompts, seeds and greedy decoding).
"""
import multiprocessing as mp
import os
import traceback
from src.models.base_vlm import Generation


def _worker(gpu, config, video, generation, tasks, results):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)  # before torch is imported in this process
    try:
        from src.models.registry import create_model
        from src.utils.reproducibility import seed_everything
        model = create_model(dict(config, device='cuda:0'), video, generation)
        model.load_model()
        results.put(('ready', gpu, model.runtime_identity, model.load_peak_allocated_bytes))
    except Exception:
        results.put(('failed', gpu, traceback.format_exc(), None))
        return
    while True:
        task = tasks.get()
        if task is None:
            return
        index, method, args, kwargs = task
        try:
            seed_everything(args[2] if method == 'generate' else args[1])
            g = getattr(model, method)(*args, **kwargs)
            results.put(('done', index, {'raw_output': g.raw_output, 'runtime': dict(g.runtime, gpu=gpu),
                                         'answer_output': g.answer_output, 'native_reasoning': g.native_reasoning}, gpu))
        except Exception:
            results.put(('error', index, traceback.format_exc(), gpu))


class BackbonePool:
    """Duck-types the backbone attributes the pipeline uses, plus generate_many()."""

    def __init__(self, gpus, config, video, generation):
        context = mp.get_context('spawn')
        self.tasks, self.results = context.Queue(), context.Queue()
        self.processes = [context.Process(target=_worker, args=(gpu, config, video, generation, self.tasks,
                                                                self.results), daemon=True) for gpu in gpus]
        for process in self.processes:
            process.start()
        identities, peaks = {}, {}
        for _ in gpus:
            status, gpu, identity, peak = self.results.get()
            if status != 'ready':
                self.close()
                raise RuntimeError(f'Backbone worker on GPU {gpu} failed to load:\n{identity}')
            identities[gpu], peaks[gpu] = identity, peak
        first = next(iter(identities.values()))
        if any(identity != first for identity in identities.values()):
            self.close()
            raise RuntimeError('Backbone workers loaded different model identities')
        self.gpus, self.runtime_identity = list(gpus), first
        self.model_id, self.generation_config, self.video_config = config['checkpoint'], generation, video
        self.load_peak_allocated_bytes = max(peaks.values())
        self.load_peaks_by_gpu = peaks

    def generate_many(self, tasks, method='generate', kwargs=None):
        """tasks: [(model_input, prompt, seed, system_prompt)] for generate, or [(prompt, seed, system_prompt)] for
        generate_text; kwargs: optional per-task keyword arguments (e.g. zoom). Results come back in order."""
        kwargs = kwargs or [{}] * len(tasks)
        for index, (task, extra) in enumerate(zip(tasks, kwargs)):
            args = task[:-1]  # system_prompt (last) goes by keyword
            self.tasks.put((index, method, tuple(args), dict(extra, system_prompt=task[-1])))
        outputs = [None] * len(tasks)
        for _ in tasks:
            status, index, payload, gpu = self.results.get()
            if status != 'done':
                raise RuntimeError(f'Generation failed on GPU {gpu}:\n{payload}')
            outputs[index] = Generation(**payload)
        return outputs

    def generate(self, model_input, prompt, seed, system_prompt=None, **kwargs):
        return self.generate_many([(model_input, prompt, seed, system_prompt)], kwargs=[kwargs])[0]

    def generate_text(self, prompt, seed, system_prompt=None, **kwargs):
        return self.generate_many([(prompt, seed, system_prompt)], method='generate_text', kwargs=[kwargs])[0]

    def close(self):
        for _ in self.processes:
            self.tasks.put(None)
        for process in self.processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
