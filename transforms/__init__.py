import utils
from datasets import BackgroundNoiseDataset
from .transform_wav import *
from torchvision.transforms import Compose


def build_transform(audio_config,
                    feature_name,
                    mode,
                    noise_path=None):
    if mode == 'train':
        if audio_config['augment']:
            augmentor = audio_config['Augmentor']
            prob = augmentor['prob']
            noise_prob = augmentor['noise_prob']
            min_shift, max_shift = augmentor['min_shift'], augmentor['max_shift'
            ]
            data_transform = Compose(
                [ChangeAmplitude(prob=prob),
                 ShiftAudio(min_shift=min_shift, max_shift=max_shift, prob=prob),
                 StretchAudio(prob=prob),
                 ChangeSpeedAndPitchAudio(prob=prob),
                 FixAudioLength(),
                 ])
        else:
            data_transform = FixAudioLength()

        if noise_path is not None:
            bg_dataset = BackgroundNoiseDataset(path=noise_path,
                                                transform=data_transform,
                                                sample_rate= audio_config['sample_rate'])
            transform = Compose([data_transform,
                                 AddBackgroundNoise(bg_dataset=bg_dataset,
                                                    prob=noise_prob)])
        else:
            transform = data_transform
    else:
        transform = FixAudioLength()

    if feature_name == 'mel_spectrogram':
        mel_spectrogram_config = audio_config['ToMelSpectrogram']
        feature_transform = Compose([
            ToMelSpectrogram(
                n_mels=mel_spectrogram_config['n_mel'],
                window=mel_spectrogram_config['window'],
                window_size=mel_spectrogram_config['window_size'],
                window_stride=mel_spectrogram_config['window_stride'],
                n_fft=mel_spectrogram_config['n_fft']),
            ToTensor('mel_spectrogram', 'input'),
        ])
    else:
        mfcc_config = audio_config['ToMFCC']
        feature_transform = Compose([
            ToMFCC(
                n_mfcc=mfcc_config['n_mfcc'],
                window=mfcc_config['window'],
                window_size=mfcc_config['window_size'],
                window_stride=mfcc_config['window_stride'],
                n_fft=mfcc_config['n_fft']),
            ToTensor('mfcc_feature', 'input'),
        ])

    return Compose([transform, feature_transform])


