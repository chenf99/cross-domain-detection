import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import warnings
import os
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import label_map, AverageMeter, save_checkpoint, clip_gradient, adjust_learning_rate
from matplotlib import pyplot as plt


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels, device)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

    return losses.avg


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--checkpoint', help='path of the pretrained model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--iteration', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    # Data parameters
    keep_difficult = True  # use objects considered difficult to detect?

    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # Model parameters
    # Not too many here since the SSD300 has a very specific structure
    n_classes = len(label_map)  # number of different types of objects
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # 防止预训练模型被加载到gpu0上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Learning parameters
    workers = 4  # number of workers for loading data in the DataLoader
    print_freq = 200  # print training status every __ batches
    decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
    momentum = 0.9  # momentum
    weight_decay = 5e-4  # weight decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True

    """
    Training.
    """

    # Initialize model or load checkpoint
    if args.checkpoint == 'pretrained_ssd300.pth.tar':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = checkpoint['model']
        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
    else:
        model = SSD300(n_classes=n_classes, device=device)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}],
                                lr=args.lr, momentum=momentum, weight_decay=weight_decay)

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(args.data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    # Calculate total number of epochs to train
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    epochs = args.iteration // (len(train_dataset) // args.batch_size)
    decay_lr_at = epochs // 2

    train_losses = []

    # Epochs
    for epoch in range(epochs):
        # Decay learning rate at particular epochs
        if epoch == decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           device=device,
                           print_freq=print_freq)
        train_losses.append(train_loss)
        # Save checkpoint
        save_checkpoint(model, os.path.join(args.result, 'model'))

    plt.title('train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(range(len(train_losses)), train_losses)
    plt.savefig(os.path.join(args.result, 'train_loss.png'))
