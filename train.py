from trainer import *
from transforms import *
from losses import *
from metrics import *
from models import *
from datasets import *

from torch.utils.data import DataLoader
import torch.optim.lr_scheduler
import argparse
import yaml
import logging


def make_logger(checkpoint_path, filename):
    logging.basicConfig(filename=checkpoint_path + '/' + filename,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    return logging.getLogger(__name__)


def get_scheduler(scheduler_cfgs, optimizer):
    scheduler_name = scheduler_cfgs['name']
    if scheduler_name == 'plateau':
        scheduler_cfgs = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_cfgs['patience'],
            factor=scheduler_cfgs['gamma']
        )

    elif scheduler_name == 'step':
        scheduler_cfgs = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfgs['step_size'],
            gamma=scheduler_cfgs['gamma'],
            last_epoch=-1
        )
    elif scheduler_name == 'cosine':
        scheduler_cfgs = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfgs['T_max'],
            last_epoch=-1
        )
    else:
        raise ValueError('Not found scheduler')
    return scheduler_cfgs, scheduler_name


def get_optimizer(opt_cfgs, model):
    name = opt_cfgs['name']
    lr = opt_cfgs['lr']
    weight_decay = opt_cfgs['weight_decay']
    if name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
    return optimizer


def set_seed(seed, use_gpu):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model for speech command',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-config_file', default='configs.yaml', type=str,
                        help='name of config file')
    parser.add_argument('-model_name', default='res15', type=str,
                        choices=['res8', 'res15', 'res26', 'res8-narrow', 'res15-narrow', 'res26-narrow'],
                        help='model name as backbone')
    parser.add_argument('-feature', type=str, default='mel_spectrogram',
                        choices=['mfcc', 'mel_spectrogram'],
                        help="type of feature input")
    parser.add_argument('-loss', type=str,
                        choices=['triplet', 'soft_triplet', 'triplet_entropy', 'cross_entropy'])
    parser.add_argument('-classify', type=bool, default=False,
                        help='optional for classify')
    parser.add_argument('-df_train', type=str, required=True,
                        help='path to train dataframe')
    parser.add_argument('-df_valid', type=str, required=True,
                        help='path to valid dataframe')
    parser.add_argument('-balance_method', type=str, default='batch_sampler',
                        choices=['batch_sampler', 'sampler'],
                        help='method for balance data')

    args = parser.parse_args()
    # np.random.seed(46)

    # Parse arguments
    config_file = args.config_file
    model_name = args.model_name
    feature = args.feature
    loss = args.loss
    classify = args.classify
    df_train = args.df_train
    df_valid = args.df_valid
    balance_method = args.balance_method

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

    background_noise_path = None
    if dataset_cfgs['add_noise']:
        background_noise_path = os.path.join(dataset_cfgs['root_dir'], dataset_cfgs['background_noise_path'])

    # Build transform
    train_transform = build_transform(audio_cfgs,
                                      mode='train',
                                      feature_name=feature,
                                      noise_path=background_noise_path)
    valid_transform = build_transform(audio_cfgs,
                                      mode='valid',
                                      feature_name=feature)

    train_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_train,
                                          audio_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=train_transform)

    valid_dataset = SpeechCommandsDataset(dataset_cfgs['root_dir'],
                                          df_valid,
                                          audio_cfgs['sample_rate'],
                                          labels=dataset_cfgs['labels'],
                                          transform=valid_transform)

    if loss == 'triplet' or loss == 'triplet_entropy':
        if loss == 'triplet':
            loss_config = param_cfgs['Loss_fn']['Triplet']
        else:
            loss_config = param_cfgs['Loss_fn']['Triplet_Entropy']

        n_samples = loss_config['samples_per_class']
        n_classes = loss_config['classes_per_batch']
        batch_size = n_classes * n_samples

        if balance_method == 'batch_sampler':
            train_batch_sampler = BalancedBatchSampler(train_dataset.get_labels('tensor'), n_classes, n_samples)
            valid_batch_sampler = BalancedBatchSampler(valid_dataset.get_labels('tensor'), n_classes, n_samples)

            # Data loader
            train_dataloader = DataLoader(train_dataset,
                                          batch_sampler=train_batch_sampler,
                                          num_workers=param_cfgs['num_workers'],
                                          pin_memory=use_gpu)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_sampler=valid_batch_sampler,
                                          num_workers=param_cfgs['num_workers'],
                                          pin_memory=use_gpu)
        else:
            train_sampler = PKSampler(train_dataset.get_labels('normal'), n_classes, n_samples)
            valid_sampler = PKSampler(valid_dataset.get_labels('normal'), n_classes, n_samples)
            # Data loader
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          sampler=train_sampler,
                                          num_workers=param_cfgs['num_workers'],
                                          pin_memory=use_gpu)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          sampler=valid_sampler,
                                          num_workers=param_cfgs['num_workers'],
                                          pin_memory=use_gpu)
    else:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=param_cfgs['batch_size'],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=param_cfgs['num_workers'],
                                      pin_memory=use_gpu)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=param_cfgs['batch_size'],
                                      shuffle=False,
                                      num_workers=param_cfgs['num_workers'],
                                      pin_memory=use_gpu)

    # Create model
    model = ResnetModel(model_name=model_name,
                        n_labels=param_cfgs['n_labels'])

    # Get optimizer, scheduler, loss function
    optimizer = get_optimizer(param_cfgs['Optimizer'], model)
    schedulers = get_scheduler(param_cfgs['Scheduler'], optimizer)
    loss_fn = make_loss_fn(param_config=param_cfgs,
                           name=loss)

    checkpoint_path = None
    if checkpoint_cfgs['resume']:
        print("Resuming a checkpoint '%s'" % checkpoint_cfgs['name'])
        # logger.info("Resuming a checkpoint '%s'" % checkpoint_cfgs['name'])
        checkpoint_path = os.path.join(checkpoint_cfgs['path'], checkpoint_cfgs['name'])

        # Create checkpoint path
    if not os.path.exists(checkpoint_cfgs['path']):
        os.mkdir(checkpoint_cfgs['path'])
    logger = make_logger(checkpoint_cfgs['path'], 'log.txt')
    logger.info('Save the config ...')
    with open(checkpoint_cfgs['path'] + '/config.yaml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)

    if loss == 'triplet_entropy':
        metric_learnings = [AccumulatedAccuracyMetric(), AverageNonzeroTripletsMetric()]
    elif loss == 'cross_entropy':
        metric_learnings = [AccumulatedAccuracyMetric()]
    else:
        metric_learnings = [AverageNonzeroTripletsMetric()]
    set_seed(46, use_gpu)
    fit(
        model=model,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        schedulers=schedulers,
        metrics=metric_learnings,
        max_epochs=param_cfgs['max_epochs'],
        start_epoch=0,
        checkpoint_path=checkpoint_path,
        save_path=checkpoint_cfgs['path'],
        use_gpu=use_gpu,
        logger=logger
    )
