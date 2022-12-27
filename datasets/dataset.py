import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import List
from utils import utils
import numpy as np


class SpeechCommandsDataset(Dataset):
    def __init__(self,
                 root_dataset_dir: str,
                 df_path: str,
                 sample_rate: int,
                 labels: List[str],
                 transform: object = None,
                 ):
        self.root_dir = root_dataset_dir
        self.df = pd.read_csv(df_path)
        self.sample_rate = sample_rate
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row['file'])
        label = utils.label_to_index(self.labels, str(row['word']))
        samples, sample_rate = utils.load_audio(path, self.sample_rate)
        data = {
            'samples': samples,
            'sample_rate': sample_rate,
            'target': label,
            'path': row['file']
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_labels(self, type_return):
        labels = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            labels.append(utils.label_to_index(self.labels, row['word']))
        if type_return == 'tensor':
            return torch.LongTensor(labels)
        else:
            return labels


class BackgroundNoiseDataset(Dataset):
    def __init__(self, path, transform, sample_rate, sample_length=1):
        noise_files = [file for file in os.listdir(path) if file.endswith('.wav')]
        samples = []
        for f in noise_files:
            noise_path = os.path.join(path, f)
            sample, sample_rate = utils.load_audio(noise_path, sample_rate)
            samples.append(sample)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {
            'samples': self.samples[index],
            'sample_rate': self.sample_rate,
            'target': 1,
            'path': self.path
        }

        if self.transform is not None:
            data = self.transform(data)
        return data
