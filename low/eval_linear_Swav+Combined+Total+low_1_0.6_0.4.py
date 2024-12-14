import sys
sys.path.append(".")
import argparse
import os
import time
from logging import getLogger
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.utils import *
# (
#     bool_flag,
#     initialize_exp,
#     restart_from_checkpoint,
#     fix_random_seeds,
#     AverageMeter,
#     init_distributed_mode,
#     accuracy,
# )
import src.resnet50V11 as resnet_models
from datasetForSIW_13 import build_dataset

logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default="./checkpoints_linear",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=True, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=400, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=16, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument("--root_dir", default="E:\data\data\SIW\dataset\SIW-13", type=str)
parser.add_argument("--mid_dir", default="z_grp_ccx/", type=str)
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")

def main():
    global args, best_acc
    args = parser.parse_args()
    # init_distributed_mode(args)/
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str('0')

    train_loader, val_loader, num_classes = \
        build_dataset(
                      batch_size=args.batch_size,
                      root_dir=args.root_dir,
                      mid_dir=args.mid_dir)
    logger.info("Building data done")

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)

    linear_classifier = RegLog(num_classes, args.arch, args.global_pooling, args.use_bn)

    # convert batch norm layers (if any)
    # linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(
    #     linear_classifier,
    #     device_ids=[args.gpu_to_work_on],
    #     find_unused_parameters=True,
    # )
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    # set optimizer
    # optimizer = torch.optim.SGD(
    #     linear_classifier.parameters(),
    #     lr=args.lr,
    #     nesterov=args.nesterov,
    #     momentum=0.9,
    #     weight_decay=args.wd,
    # )

    optimizer = torch.optim.Adam(
        linear_classifier.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint_low34_all_75_1_0.6_0.4.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        # train_loader.sampler.set_epoch(epoch)

        # scores = train(model, linear_classifier, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model, linear_classifier)
        # training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint_low34_all_75_1_0.6_0.4.pth.tar"))
    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            # if arch == "resnet50":
            #     s = 512
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):

        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch, classification=True):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()

    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()
    for i in range(0, len(loader)):

        for iter_epoch, (inp, target) in enumerate(loader[i]):
            # measure data loading time
            # data_time.update(time.perf_counter() - end)

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # forward
            with torch.no_grad():
                output = model(inp, classification)

            # output[0] torch.Size([4, 2048, 5, 9])
            # output[1] torch.Size([4, 2048, 2, 2])
            output = reglog(output)

            # compute cross entropy loss
            loss = criterion(output, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            # update stats
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # verbose
            if args.rank == 0 and iter_epoch % 50 == 0:
                logger.info(
                    "Epoch[{0}] - Iter: [{1}/{2}]\t"
                    # "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    # "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                    "LR {lr}".format(
                        epoch,
                        iter_epoch,
                        len(loader[i]),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, linear_classifier, classification=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    conf_matrix = np.zeros([13, 13])

    with torch.no_grad():
        end = time.perf_counter()
        for i in range(0, len(val_loader)):

            for j, (inp, target) in enumerate(val_loader[i]):

                # move to gpu
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                output = linear_classifier(model(inp, classification))
                loss = criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), inp.size(0))
                top1.update(acc1[0], inp.size(0))
                top5.update(acc5[0], inp.size(0))

                _, predictions = torch.max(output, 1)

                for i in range(len(target)):
                    conf_matrix[target[i], predictions[i]] += 1


    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    if args.rank == 0:
        logger.info(
            "Test:\t"
            # "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))
    classes = ('Arabic', 'Cambodian', 'Chinese', 'English', 'Greek', 'Hebrew', 'Japanese', 'Kannada',
               'Korean', 'Mongolian', 'Russian', 'Thai', 'Tibetan')
    plot_confusion_matrix(conf_matrix, list(classes), "Confusion_Matrix_siw.jpeg", "SIW-13")

    return losses.avg, top1.avg.item(), top5.avg.item()



def plot_confusion_matrix(cm, classes, savename, title, normalize=False, cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm.astype('float')

        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j]), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    plt.savefig(savename)
    plt.show()

if __name__ == "__main__":
    main()