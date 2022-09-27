from .resnet import *


def get_model(model_name: str):
    if model_name.startswith('res'):
        return SpeechResModel(resnet_configs[model_name])
    else:
        raise ValueError('Not found the model %s' % model_name)