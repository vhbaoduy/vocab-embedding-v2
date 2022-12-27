import argparse

import pandas as pd

from transforms import *
from torch.utils.data import DataLoader
from metrics import *
from models import *
from datasets import *
from tqdm import tqdm
import os
import numpy as np


def do_inference(model,
                 data_loader,
                 metric,
                 path,
                 labels,
                 use_gpu,
                 device):
    with torch.no_grad():
        metric.reset()
        counter = {}
        model.eval()
        pbar = tqdm(data_loader, desc='Test: ')
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            preds, feat = model(inputs)
            metric(preds, targets, None)

            ordered_dict = {metric.name(): metric.value()}
            pbar.set_postfix(ordered_dict)

            if preds is not None:
                prediction = preds.data.max(1, keepdim=True)[0].cpu().numpy().ravel()

            # Convert gpu to cpu
            # preds = preds.cpu().numpy().ravel()
            targets = targets.cpu().numpy().ravel()
            if feat is not None:
                feat = feat.cpu().numpy()

            if save_hit:
                cur_batch_size = len(batch['path'])
                for i in range(cur_batch_size):
                    # file_name = batch['path'][i]
                    name_class = utils.index_to_label(labels, str(prediction[i]))
                    truth_class = utils.index_to_label(labels, str(targets[i]))

                    if name_class not in counter:
                        counter[name_class]['true'] = 0
                        counter[name_class]['false'] = 0
                    
                    if name_class == truth_class:
                        counter[name_class]['true'] += 1
                    else:
                        counter[name_class]['false'] += 1

        data = {
            'word': [],
            'true': [],
            'false': [],
            'accuracy': []
        }
        for key in counter:
            data['word'].append(key)
            t = counter[key]['true']
            f = counter[key]['false']
            data['true'].append(t)
            data['f'].append(f)
            data['accuracy'].append(t / (t+f))
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str, help='name of config file')
    parser.add_argument('-model_name', default='res15', type=str,
                        choices=['res8', 'res15', 'res26', 'res8-narrow', 'res15-narrow', 'res26-narrow'],
                        help='model name as backbone')
    parser.add_argument('-model_path', type=str, help='path to model')
    parser.add_argument('-embedding_size', type=int, default=128, help="dimension of embeddings")
    parser.add_argument('-feature', type=str, default='mel_spectrogram',
                        choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")
    parser.add_argument('-path_to_df', type=str, help='path to dataframe')
    parser.add_argument('-batch_size', type=int, default=128,
                        help="batch size for inference")
    parser.add_argument('-path', type=str, default='./inferences',
                        help="path to store features")
    parser.add_argument('-mode', type=str,
                        choices=['truth', 'predict', 'intersect'],
                        help='use ground truth')
    parser.add_argument('-classify', type=bool, default=False,
                        help='optional for classify')
    args = parser.parse_args()

    # Parse arguments
    config_file = args.config_file
    model_name = args.model_name
    embedding_size = args.embedding_size
    feature = args.feature
    model_path = args.model_path
    path = args.path
    batch_size = args.batch_size
    path_to_df = args.path_to_df
    mode = args.mode
    classify = args.classify

    # Load configs
    configs = utils.load_config_file(os.path.join('./configs', config_file))
    dataset_cfgs = configs['Dataset']
    audio_cfgs = configs['AudioProcessing']
    param_cfgs = configs['Parameters']
    checkpoint_cfgs = configs['Checkpoint']
    device = param_cfgs['device']

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    labels = dataset_cfgs['labels']
    # Load dataframe
    test_transform = build_transform(audio_cfgs,
                                     mode='valid',
                                     feature_name=feature)
    test_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                         path_to_df,
                                         audio_cfgs['sample_rate'],
                                         labels=dataset_cfgs['labels'],
                                         transform=test_transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=param_cfgs['num_workers'],
                                 pin_memory=use_gpu)

    model = ResnetModel(model_name=model_name,
                        n_labels=len(labels))

    model = torch.load(model_path, map_location=torch.device(device))
    # model.load(model_path)
    if use_gpu:
        # model = nn.DataParallel(model).cuda()
        model = model.to(device)

    metric_learning = AccumulatedAccuracyMetric()
    if not os.path.exists(path):
        os.mkdir(path)

    do_inference(model=model,
                 data_loader=test_dataloader,
                 metric=metric_learning,
                 path=path,
                 labels=labels,
                 use_gpu=use_gpu,
                 device=device)
