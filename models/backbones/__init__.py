from .resnet import *


def get_model(model_name: str,
              n_labels: int):
    if model_name.startswith('res'):
        return SpeechResModel(resnet_configs[model_name], n_labels)
    else:
        raise ValueError('Not found the model %s' % model_name)