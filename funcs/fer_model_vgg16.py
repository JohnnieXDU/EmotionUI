import torch.nn as nn
import math
import torch

from torchvision.transforms import functional as F


class vgg16face(nn.Module):
    mean = [0.5830, 0.4735, 0.4262]
    std = [0.2439, 0.1990, 0.1819]

    def __init__(self, weights_dir="random", cuda=True):
        super(vgg16face, self).__init__()

        self.cuda = cuda

        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_2_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_3_2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_3_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4_1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_4_2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_4_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_5_2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_5_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7),
        )

        self._initialize_weights(weights_dir)

    def forward(self, x):
        # normalization
        x = F.normalize(x, self.mean, self.std)

        # forward
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.pool_1(x)  # [bs, 64, 112, 112]

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.pool_2(x)  # [bs, 128, 56, 56]

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.pool_3(x)  # [bs, 256, 28, 28]

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.pool_4(x)  # [bs, 512, 14, 14]

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.pool_5(x)  # [bs, 512, 7, 7]

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, weights_dir="random"):
        if weights_dir == "random":  # random init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
        else:
            if self.cuda:
                weights = torch.load(weights_dir)
            else:
                weights = torch.load(weights_dir, map_location=torch.device('cpu'))
            self.load_state_dict(weights)


# ======================== Define Net Functions =================== #
def vgg16face_net(custom_pretrained=False, custom_weights_dir='', random_init=False):
    model = vgg16face()
    if not random_init:
        if custom_pretrained:
            # 2). Trained on Face Emotion
            weight = torch.load(custom_weights_dir)
            model.load_state_dict(weight)
        else:
            # 1). PyTorch Official - pretrained weights
            official_weights = '/home/server109/JY/Model_Zoo_JY/vgg16face-jy.pth'
            pretrained_dict = torch.load(official_weights)
            model_dict = model.state_dict()

            same_layer = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(same_layer)
            model.load_state_dict(model_dict)

    return model
