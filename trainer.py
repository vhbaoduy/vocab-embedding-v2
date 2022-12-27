from datasets import *
from metrics import *
from models import *
from tqdm import tqdm
import time

text_epoch = ''


def _get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def fit(model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        schedulers,
        max_epochs,
        metrics,
        logger=None,
        use_gpu=True,
        device='cpu',
        start_epoch=0,
        checkpoint_path=None,
        save_path=None):
    if use_gpu:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.to(device)

    if logger is not None:
        logger.info("Start Training ...")
    best_loss = 1e100
    best_acc = 0
    scheduler, scheduler_name = schedulers

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_acc = checkpoint.get('accuracy', best_acc)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)

    for i in range(0, start_epoch):
        if scheduler_name != "plateau":
            scheduler.step()

    since = time.time()
    for epoch in range(start_epoch, max_epochs):
        print("Epoch ", epoch)
        global text_epoch
        text_epoch = "Epoch %s" % epoch
        train_loss, metrics = do_train(model, train_loader, optimizer, loss_fn, metrics, use_gpu,device=device)
        valid_loss, metrics = do_validation(model, val_loader, loss_fn, metrics, use_gpu,device=device)
        if scheduler_name == "plateau":
            scheduler.step(metrics=valid_loss)

        if scheduler_name != "plateau":
            scheduler.step()
        acc = 0
        for metric in metrics:
            if metric.name() == 'Accuracy':
                acc = metric.value()
        save_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'loss': valid_loss,
            'optimizer': optimizer.state_dict(),
            'accuracy': acc
        }
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(save_checkpoint, save_path + '/' + 'best-loss.pth')
            torch.save(model, save_path + '/' + 'best-loss-model.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(save_checkpoint, save_path + '/' + 'best-acc.pth')
            torch.save(model, save_path + '/' + 'best-acc-model.pth')
        torch.save(save_checkpoint, save_path + '/' + 'last-checkpoint.pth')

        time_elapsed = time.time() - since
        time_str = 'Total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        text_epoch += "\n%s, Best loss %f, Best accuracy %.02f%%" % (time_str, best_loss, best_acc)
        print("%s, Best loss %f, Best accuracy %.02f%%" % (time_str, best_loss, best_acc))
        if logger is not None:
            logger.info(text_epoch)

    if logger is not None:
        logger.info('Finished ...')


def do_train(model,
             train_loader,
             optimizer,
             loss_fn,
             metrics,
             use_gpu=False,
             device='cpu'):
    for metric in metrics:
        metric.reset()
    global text_epoch
    model.train()
    losses = []
    total_loss = 0
    pbar = tqdm(train_loader, desc='Train: ')
    for batch in pbar:
        inputs = batch['input']
        targets = batch['target']

        if use_gpu:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        preds, feat = model(inputs)

        loss = loss_fn(preds, feat, targets)
        losses.append(loss[0].item())
        total_loss += loss[0].item()

        loss[0].backward()
        optimizer.step()
        for metric in metrics:
            metric(preds, targets, loss)

        ordered_dict = {'lr': _get_lr(optimizer), 'Loss': np.mean(losses)}
        for metric in metrics:
            ordered_dict[metric.name()] = metric.value()

        pbar.set_postfix(ordered_dict)
    text_epoch += "\nTrain loss: " + str(np.mean(losses)) + " "
    for metric in metrics:
        text_epoch += metric.name() + ': ' + str(metric.value()) + " "
    # logger.info(text)
    return np.mean(losses), metrics


def do_validation(model,
                  val_loader,
                  loss_fn,
                  metrics,
                  use_gpu,
                  device='cpu'):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        global text_epoch
        model.eval()
        losses = []
        val_loss = 0
        pbar = tqdm(val_loader, desc='Validate: ')
        for batch in pbar:
            inputs = batch['input']
            targets = batch['target']

            if use_gpu:
                inputs = inputs.to(device)
                targets = targets.to(device)

            preds, feat = model(inputs)
            loss = loss_fn(preds, feat, targets)

            losses.append(loss[0].item())
            val_loss += loss[0].item()

            for metric in metrics:
                metric(preds, targets, loss)

            ordered_dict = {'Loss': np.mean(losses)}
            for metric in metrics:
                ordered_dict[metric.name()] = metric.value()

            pbar.set_postfix(ordered_dict)
        text_epoch += "\nValid loss: " + str(np.mean(losses)) + " "
        for metric in metrics:
            text_epoch += metric.name() + ': ' + str(metric.value()) + " "
        return np.mean(losses), metrics


if __name__ == '__main__':
    pass
