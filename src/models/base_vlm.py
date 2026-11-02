from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.datasets.video_loader import load_video, load_frames


@dataclass
class Generation:
    raw_output: str
    runtime: dict = field(default_factory=dict)
    answer_output: str | None = None
    native_reasoning: str | None = None


class BaseVLM(ABC):
    def __init__(self, config, video, generation):
        self.config, self.video_config, self.generation_config = config, video, generation
        self.model_id = config['checkpoint']
        self.runtime_identity = dict(config)

    @abstractmethod
    def load_model(self): ...

    def preprocess_video(self, path):
        config = {k: v for k, v in self.video_config.items() if k != 'assumed_sampled_fps'}
        return load_video(path, **config)

    def preprocess_media(self, model_input):
        if model_input.get('frame_paths'):
            return load_frames(model_input['frame_paths'], **self.video_config)
        return self.preprocess_video(model_input['video_path'])

    @staticmethod
    def chat_messages(user_content, system_prompt=None):
        # Without a system prompt the message list is unchanged from earlier runs.
        system = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}] if system_prompt else []
        return system + [{'role': 'user', 'content': user_content}]

    @abstractmethod
    def generate(self, model_input, prompt, seed, system_prompt=None): ...

    @abstractmethod
    def unload_model(self): ...
