from .triplet_loss import *
from .soft_triplet import *
from .triplet_selector import *


def make_loss_fn(param_config,
                 name):
    loss_fn_config = param_config['Loss_fn']
    if name == 'triplet' or name == 'triplet_entropy':
        if name == 'triplet':
            config = loss_fn_config['Triplet']
        else:
            config = loss_fn_config['Triplet_Entropy']
        selector = config['selector']
        if selector == 'hardest':
            triplet_selector = HardestNegativeTripletSelector(config['margin'])
        elif selector == 'semi_hard':
            triplet_selector = SemihardNegativeTripletSelector(config['margin'])
        elif selector == 'random':
            triplet_selector = RandomNegativeTripletSelector(config['margin'])
        elif selector == 'all':
            triplet_selector = AllTripletSelector()
        else:
            raise ValueError('Not found triplet selector')
        triplet_loss = OnlineTripletLoss(config['margin'], triplet_selector)

        if name == 'triplet':
            def loss_fn(scores, embeddings, target):
                return triplet_loss(embeddings, target)
        else:
            alpha = config['alpha']
            beta = config['beta']

            def loss_fn(scores, embeddings, target):
                loss_A, n_triplets = triplet_loss(embeddings, target)
                loss_B = F.cross_entropy(scores, target)
                return alpha * loss_A + beta * loss_B, n_triplets

    elif name == 'soft_triplet':
        config = loss_fn_config['SoftTriplet']
        soft_triplet_loss = SoftTripleLoss(param_config['n_labels'],
                                           param_config['embedding_size'],
                                           config['centers_per_class'],
                                           config['lambda'],
                                           config['gamma'],
                                           config['tau'],
                                           config['margin'],
                                           param_config['device'])

        def loss_fn(scores, embeddings, target):
            return soft_triplet_loss(embeddings, target)
    elif name == 'cross_entropy':
        def loss_fn(scores, embeddings, target):
            loss = nn.CrossEntropyLoss()
            return loss(scores, target), 0
    else:

        raise ValueError('Not found loss function')
    return loss_fn
