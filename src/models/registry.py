from src.models.qwen35_vl import Qwen35VL

MODELS = {'qwen35_vl': Qwen35VL}


def create_model(config, video, generation):
    return MODELS[config['adapter']](config, video, generation)
