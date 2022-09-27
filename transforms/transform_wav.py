import numpy as np
import librosa
import random
import torch

import utils


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data


class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1), prob=0.5):
        self.amplitude_range = amplitude_range
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        samples = data['samples']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0 / (1 + scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0, len(samples)), samples).astype(
            np.float32)
        return data


class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(y=data['samples'], rate=1 + scale)
        return data


class ShiftAudio(object):
    """Shift audio randomly """

    def __init__(self, min_shift, max_shift, prob=0.5):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        sample_rate = data['sample_rate']
        shift_range = int(np.random.uniform(self.min_shift * sample_rate / 1000, self.max_shift * sample_rate / 1000))
        # print(shift_range)
        data['samples'] = np.roll(data['samples'], shift_range)
        return data


class AddBackgroundNoise(object):
    """ Add background noise to wave"""

    def __init__(self, bg_dataset, max_percentage=0.45, prob=0.5):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data


class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a n_mels x 32 matrix."""

    def __init__(self,
                 n_mels=32,
                 n_fft=512,
                 window='hann',
                 window_size=0.025,
                 window_stride=0.01):
        self.window = window
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.window_size = window_size
        self.window_stride = window_stride

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        s = librosa.feature.melspectrogram(y=samples,
                                           sr=sample_rate,
                                           n_mels=self.n_mels,
                                           window=self.window,
                                           win_length=int(self.window_size * sample_rate),
                                           hop_length=int(self.window_stride * sample_rate),
                                           n_fft=self.n_fft)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data


class ToMFCC(object):
    def __init__(self,
                 n_mfcc=40,
                 window_size=0.025,
                 window_stride=0.01,
                 n_fft=512,
                 window='hann'):
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_fft = n_fft
        self.window = window

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        mfcc_feature = librosa.feature.mfcc(y=samples,
                                            sr=sample_rate,
                                            n_mfcc=self.n_mfcc,
                                            window=self.window,
                                            win_length=int(self.window_size * sample_rate),
                                            hop_length=int(self.window_stride * sample_rate),
                                            n_fft=self.n_fft)
        data['mfcc_feature'] = mfcc_feature
        return data


class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data


# debug
if __name__ == '__main__':
    from scipy.io.wavfile import write

    np.random.seed(10)
    path = 'F:\\Datasets\\speech_commands_v0.02\\backward'
    file = '0a2b400e_nohash_3.wav'
    samples, rate = utils.load_audio(path + '\\' + file, 16000)
    data = {
        'samples': samples,
        "sample_rate": rate
    }
    data = ShiftAudio(-100, 100, 1.2)(data)
    write('temp.wav', 16000, samples)
    # data = {
    #     'samples': samples,
    #     'sample_rate': rate
    # }
    # data = ToMFCC()(data)
    # print(data['mfcc_feature'].shape)
