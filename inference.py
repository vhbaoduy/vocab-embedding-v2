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
                 save_hit,
                 path,
                 labels,
                 mode,
                 use_gpu):
    with torch.no_grad():
        metric.reset()
        hits = {}
        counter = {}
        model.eval()
        pbar = tqdm(data_loader, desc='Test: ')
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            if use_gpu:
                inputs = inputs.to('cuda')
                targets = targets.to('cuda')

            preds, feat = model(inputs)
            metric(preds, targets, None)

            ordered_dict = {metric.name(): metric.value()}
            pbar.set_postfix(ordered_dict)

            if preds is not None:
                prediction = preds.data.max(1, keepdim=True)
                prediction = prediction[0].cpu().numpy().ravel()
                probs = prediction[1].cpuu().numpy().ravel()

            # Convert gpu to cpu
            # preds = preds.cpu().numpy().ravel()
            targets = targets.cpu().numpy().ravel()
            if feat is not None:
                feat = feat.cpu().numpy()

            if save_hit:
                cur_batch_size = len(batch['path'])
                for i in range(cur_batch_size):
                    # file_name = batch['path'][i]
                    name_class = utils.index_to_label(labels, prediction[i])
                    truth_class = utils.index_to_label(labels, targets[i])
                    # folder = os.path.join(path, truth_class)
                    #
                    # if not os.path.exists(folder):
                    #     os.mkdir(folder)
                    # audio_name = file_name.split('/')[1].split('.')[0]

                    if truth_class in hits:
                        if truth_class == name_class:
                            hits[truth_class].append(probs[i])
                            counter[truth_class] += 1
                        else:
                            hits[truth_class].append(0)
                    else:
                        if truth_class == name_class:
                            hits[truth_class] = [probs[i]]
                            counter[truth_class] = 1
                        else:
                            hits[truth_class] = [0]

        data = {
            'vocab': [],
            'prob_accumulation': [],
            'hit': [],
            'total': []
        }
        for key in hits:
            total = sum(hits[key])
            cnt = counter[key]
            data['vocab'].append(key)
            data['prob_accumulation'].append(total)
            data['hit'].append(cnt)
            data['total'].append(len(hits[key]))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(path, 'result.csv'), index=False)
        return hits, counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str, help='name of config file')
    parser.add_argument('-model_name', default='resnet15', type=str,
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
    parser.add_argument('-save_hit', type=bool, required=True,
                        help="optional")
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
    save_hit = args.save_hit
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

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    labels = dataset_cfgs['labels']
    # Load dataframe
    df_test = pd.read_csv(path_to_df)
    test_transform = build_transform(audio_cfgs,
                                     mode='valid',
                                     feature_name=feature)
    test_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                         df_test,
                                         audio_cfgs['sample_rate'],
                                         labels=dataset_cfgs['labels'],
                                         transform=test_transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=param_cfgs['num_workers'],
                                 pin_memory=use_gpu)

    model = ResnetModel(model_name=model_name,
                        n_labels=len(labels))

    # model = torch.load(model_path, map_location=torch.device('cpu'))
    model.load(model_path)
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    metric_learning = AccumulatedAccuracyMetric()
    if not os.path.exists(path):
        os.mkdir(path)

    do_inference(model=model,
                 data_loader=test_dataloader,
                 metric=metric_learning,
                 save_hit=save_hit,
                 path=path,
                 labels=labels,
                 mode=mode,
                 use_gpu=use_gpu)
