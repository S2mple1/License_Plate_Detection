import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, trt=False):
        super(myNet_ocr, self).__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        self.trt = trt
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.loc(x)
        x = self.newCnn(x)
        if self.export:
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.transpose(2, 1)  # [w, b, c]
            if self.trt:
                conv = conv.argmax(dim=2)
                conv = conv.float()
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            output = F.log_softmax(conv, dim=2)

            return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)
    out = model(x)
    print(out.shape)
