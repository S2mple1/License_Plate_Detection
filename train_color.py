import torch
import torch.nn as nn
from torch.autograd import Variable
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
import numpy as np
import os
import argparse
from Net.colorNet import myNet_ocr_color
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

train_loss_list = []
val_loss_list = []
accuracy_list = []


def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.num_features != 5 and m.num_features != 12:
            m.eval()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    print(scheduler.get_lr())
    model.train()
    model.apply(fix_bn)

    epoch_loss = 0.0
    total_batches = len(trainloader)

    for batch_idx, (img, label) in enumerate(tqdm(trainloader, desc=f'Training Epoch {epoch}')):
        image = Variable(img.cuda())
        label = Variable(label.cuda())
        optimizer.zero_grad()
        _, out = model(image)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 50 == 0:
            print("Epoch:%d [%d|%d] loss:%f lr:%s" % (
                epoch, batch_idx, total_batches, loss.mean(), scheduler.get_lr()))

    avg_epoch_loss = epoch_loss / total_batches
    train_loss_list.append(avg_epoch_loss)
    print(f"Avg Loss for Epoch {epoch}: {avg_epoch_loss}")

    scheduler.step()


def val(epoch):
    print("\nValidation Epoch: %d" % epoch)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valloader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            _, out = model(image)
            _, predicted = torch.max(out.data, 1)
            total += image.size(0)
            correct += predicted.data.eq(label.data).cuda().sum()
    accuracy = 1.0 * correct.cpu().numpy() / total
    accuracy_list.append(accuracy)
    print("Acc: %f " % ((1.0 * correct.cpu().numpy()) / total))
    exModelName = opt.model_path + '/' + str(format(accuracy, ".6f")) + "_" + "epoch_" + str(epoch) + "_model" + ".pth"
    torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, exModelName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='weights/plate_rec_ocr.pth')  # 车牌识别模型
    parser.add_argument('--train_path', type=str, default='datasets/plate_color/train')  # 颜色训练集
    parser.add_argument('--val_path', type=str, default='datasets/plate_color/val')  # 颜色验证集
    parser.add_argument('--num_color', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='color_model', help='model_path')
    opt = parser.parse_args()

    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.device == 'cuda' else 'cpu')
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)

    mean_value = (0.588, 0.588, 0.588)
    std_value = (0.193, 0.193, 0.193)

    transform_train = cvTransforms.Compose([
        cvTransforms.Resize((48, 168)),
        cvTransforms.RandomHorizontalFlip(),
        cvTransforms.ToTensorNoDiv(),
        cvTransforms.NormalizeCaffe(mean_value, std_value)
    ])

    transform_val = cvTransforms.Compose([
        cvTransforms.Resize((48, 168)),
        cvTransforms.ToTensorNoDiv(),
        cvTransforms.NormalizeCaffe(mean_value, std_value),
    ])

    rec_model_Path = opt.weights  # 车牌识别模型
    checkPoint = torch.load(rec_model_Path, map_location=torch.device('cuda' if opt.device == 'cuda' else 'cpu'))
    cfg = checkPoint["cfg"]
    print(cfg)
    model = myNet_ocr_color(cfg=cfg, color_num=opt.num_color)
    model_dict = checkPoint['state_dict']
    model.load_state_dict(model_dict, strict=False)

    trainset = dset.ImageFolder(opt.train_path, transform=transform_train, loader=cv_imread)
    valset = dset.ImageFolder(opt.val_path, transform=transform_val, loader=cv_imread)
    print(len(valset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True,
                                              num_workers=opt.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batchSize, shuffle=False,
                                            num_workers=opt.num_workers)
    model = model.to(device)
    for name, value in model.named_parameters():
        if name not in ['color_classifier.weight', 'color_classifier.bias', 'color_bn.weight', 'color_bn.bias',
                        'conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias']:
            value.requires_grad = False
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.epoch))
    criterion = CrossEntropyLabelSmooth(opt.num_color)
    criterion.cuda()
    for epoch in range(opt.epoch):
        train(epoch)
        val(epoch)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(opt.model_path, 'training_curves.png'))
    plt.close()
