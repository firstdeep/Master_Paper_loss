import torch
import torch.nn as nn
from torch.nn import init


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.Conv1 = conv_block(ch_in=1, ch_out=64)
        # self.Conv2 = conv_block(ch_in=64, ch_out=128)
        # self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        # self.Up3 = up_conv(ch_in=256, ch_out=128)
        # self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        #
        # self.Up2 = up_conv(ch_in=128, ch_out=64)
        # self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1 = nn.Conv2d(256, output_ch, kernel_size=1, stride=1, padding=0)

    # def forward(self, x):
    #     # encoding path
    #     x1 = self.Conv1(x)
    #     # print("conv1 shape", x1.size())
    #     x2 = self.Maxpool(x1)
    #     # print("maxpool1 shape", x2.size())
    #     x2 = self.Conv2(x2)
    #     # print("conv2 shape", x2.size())
    #
    #     x3 = self.Maxpool(x2)
    #     # print("maxpool2 shape", x3.size())
    #     x3 = self.Conv3(x3)
    #     # print("conv3 shape", x3.size())
    #
    #     x4 = self.Maxpool(x3)
    #     # print("maxpool3 shape", x4.size())
    #     x4 = self.Conv4(x4)
    #     # print("conv4 shape", x4.size())
    #
    #     x5 = self.Maxpool(x4)
    #     # print("maxpool4 shape", x5.size())
    #     x5 = self.Conv5(x5)
    #     # print("conv5 shape", x5.size())
    #
    #     # decoding + concat path
    #     print("========================================")
    #     d5 = self.Up5(x5)
    #     # print("deconv shape", d5.size())
    #     d5 = torch.cat((x4, d5), dim=1)
    #     print("========================================")
    #
    #     d5 = self.Up_conv5(d5)
    #
    #     d4 = self.Up4(d5)
    #     d4 = torch.cat((x3, d4), dim=1)
    #     d4 = self.Up_conv4(d4)
    #
    #     d3 = self.Up3(d4)
    #     d3 = torch.cat((x2, d3), dim=1)
    #     d3 = self.Up_conv3(d3)
    #
    #     d2 = self.Up2(d3)
    #     d2 = torch.cat((x1, d2), dim=1)
    #     d2 = self.Up_conv2(d2)
    #
    #     d1 = self.Conv_1x1(d2)
    #
    #     return d1

    def forward(self, x):

        x4 = self.Conv4(x)
        # print("conv4 shape", x4.size())

        x5 = self.Maxpool(x4)
        # print("maxpool4 shape", x5.size())
        x5 = self.Conv5(x5)
        # print("conv5 shape", x5.size())

        # decoding + concat path
        print("========================================")
        d5 = self.Up5(x5)
        # print("deconv shape", d5.size())
        d5 = torch.cat((x4, d5), dim=1)
        print("========================================")

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d1 = self.Conv_1x1(d4)

        return d1


if __name__ =="__main__":
    x = torch.zeros((10,256,112,112))
    model = U_Net(img_ch=1, output_ch=2)

    output = model(x)
    print(output.shape)
    print("===")