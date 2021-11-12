# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from model import CNN, Unet
from nni.nas.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch

logger = logging.getLogger('nni')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def train(config, train_loader, model, optimizer, criterion, epoch):
    # top1 = AverageMeter("top1")
    # top5 = AverageMeter("top5")
    losses = AverageMeter("losses")
    dice = AverageMeter("dice")
    mIoU = AverageMeter("mIoU")
    dice.reset()
    mIoU.reset()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)
        y = torch.squeeze(y).long()

        optimizer.zero_grad()
        # logits, aux_logits = model(x)
        logits = model(x)
        loss = criterion(logits, y)
        # if config.aux_weight > 0.:
        #     loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # accuracy = utils.accuracy(logits, y, topk=(1, 5))
        accuracy = utils.accuracy(logits, y)
        mIoU.update(utils.mIoU(logits, y, config.nclasses).item(), 1)
        dice.update(utils.dice_coefficient(logits, y).item(), 1)
        losses.update(loss.item(), bs)
        # top1.update(accuracy["acc1"], bs)
        # top5.update(accuracy["acc5"], bs)

        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        # writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        # writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)
        writer.add_scalar("acc/train", accuracy, global_step=cur_step)
        writer.add_scalar("mIoU/train", mIoU.avg, global_step=cur_step)
        writer.add_scalar("dice/train", dice.avg, global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            # logger.info(
            #     "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
            #     "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
            #         epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
            #         top1=top1, top5=top5))

            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Acc {acc:.4%} mIOU {miou.avg:.4%} dice {dice.avg:.4%}".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    acc=accuracy, miou=mIoU, dice=dice))

        cur_step += 1

    # logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    logger.info("Train: [{:3d}/{}] Final acc={:.4%} mIoU={:.4%} dice={:.4%}".format(epoch + 1, config.epochs, accuracy,
                                                                                    mIoU.avg, dice.avg))


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    # top1 = AverageMeter("top1")
    # top5 = AverageMeter("top5")
    losses = AverageMeter("losses")
    dice = AverageMeter("dice")
    mIoU = AverageMeter("mIoU")
    mIoU.reset()
    dice.reset()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)
            y = torch.squeeze(y).long()

            logits = model(X)
            loss = criterion(logits, y)

            # accuracy = utils.accuracy(logits, y, topk=(1, 5))
            accuracy = utils.accuracy(logits, y)
            mIoU.update(utils.mIoU(logits, y, config.nclasses).item(), 1)
            dice.update(utils.dice_coefficient(logits, y).item(), 1)
            losses.update(loss.item(), bs)
            # top1.update(accuracy["acc1"], bs)
            # top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                # logger.info(
                #     "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                #     "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                #         epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                #         top1=top1, top5=top5))
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc ({acc:.4%}) mIoU {mIoU.avg:.4%} dice {dice.avg:.4%}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        acc=accuracy, mIoU=mIoU, dice=dice))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    # writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    # writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)
    writer.add_scalar("acc/test", accuracy, global_step=cur_step)
    writer.add_scalar("mIoU/test", mIoU.avg, global_step=cur_step)
    writer.add_scalar("dice/test", dice.avg, global_step=cur_step)

    # logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    logger.info("Valid: [{:3d}/{}] Final acc={:.4%} mIoU={:.4%} dice ={:.4%}".format(epoch + 1, config.epochs, accuracy
                                                                                     , mIoU.avg, dice.avg))

    # return top1.avg
    return accuracy, mIoU.avg, dice.avg

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=0)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./final architecture/20210812-165458checkpoint.json")

    args = parser.parse_args()
    # dataset_train, dataset_valid = datasets.get_dataset("cifar10", cutout_length=16)
    dataset_train, dataset_valid = datasets.get_dataset("brats2015", cutout_length=0)

    with fixed_arch(args.arc_checkpoint):
        # model = CNN(32, 3, 36, 10, args.layers, auxiliary=True)
        model = Unet(in_channels=4, n_classes=5, n_layers=3, n_nodes=4)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    # best_top1 = 0.
    best_acc, best_mIoU, best_dice = 0., 0., 0.
    for epoch in range(args.epochs):
        # drop_prob = args.drop_path_prob * epoch / args.epochs
        # model.drop_path_prob(drop_prob)

        # training
        train(args, train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        # top1 = validate(args, valid_loader, model, criterion, epoch, cur_step)
        # best_top1 = max(best_top1, top1)
        accuracy, mIoU, dice = validate(args, valid_loader, model, criterion, epoch, cur_step)
        best_acc = max(best_acc, accuracy)
        best_mIoU = max(best_mIoU, mIoU)
        best_dice = max(best_dice, dice)

        lr_scheduler.step()

    # logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Final best acc = {:.4%} mIoU = {:.4%} dice={:.4%}".format(best_acc, best_mIoU, best_dice))
    torch.save(model, './final model/checkpoint_{}.json'.format(time.strftime('%Y%m%d-%H%M%S')))
