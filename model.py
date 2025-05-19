import torch.nn as nn
import torch

import MinkowskiEngine as ME


class UpSampleAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.v_mlp = nn.Sequential(
            ME.MinkowskiLinear(in_c, in_c),
            ME.MinkowskiReLU(),
        )
        self.k_mlp = nn.Sequential(
            ME.MinkowskiLinear(in_c, in_c),
            ME.MinkowskiReLU(),
        )
        self.q_mlp = nn.Sequential(
            ME.MinkowskiLinear(in_c, in_c),
            ME.MinkowskiReLU(),
        )
        self.rse = nn.Sequential(
            ME.MinkowskiLinear(in_c, in_c),
            ME.MinkowskiBatchNorm(in_c),
            ME.MinkowskiGELU(),
            ME.MinkowskiLinear(in_c, in_c),
            ME.MinkowskiSigmoid()
        )

    def forward(self, x1, x2):
        x1_v = self.v_mlp(x1)
        x1_k = self.k_mlp(x1)
        x2_q = self.q_mlp(x2)
        x1_rse = self.rse(x1)

        a = x1_k * x2_q
        f = a * x1_v
        m = f * x1_rse

        out_ = f + m

        return out_


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(inc, outc, kernel_size=ks, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),

            ME.MinkowskiConvolution(outc, outc, kernel_size=ks, dilation=dilation, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(outc)
        )

        if inc == outc and stride == 1:
            self.down_sample = nn.Sequential()
        else:
            self.down_sample = nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc)
            )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.down_sample(x))
        return out


class OursModel(nn.Module):
    def __init__(self, in_channels, d_no_what=3, is_cat=False):
        super().__init__()

        self.in_channels = in_channels
        cs = [8, 32, 128, 256, 512, 256, 128, 32, 8]
        self.D = d_no_what
        self.is_cat = is_cat

        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.mlp = ME.MinkowskiLinear(cs[4], cs[4])

        self.up_sample1 = BasicDeconvolutionBlock(cs[5], cs[5], ks=8, stride=8, D=self.D)
        self.up_sample2 = BasicDeconvolutionBlock(cs[6], cs[6], ks=4, stride=4, D=self.D)
        self.up_sample3 = BasicDeconvolutionBlock(cs[7], cs[7], ks=2, stride=2, D=self.D)

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
            nn.Sequential(ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1, D=self.D),
                          ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D))
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
            nn.Sequential(ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1, D=self.D),
                          ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D))
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
            nn.Sequential(ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1, D=self.D),
                          ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D))
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
            nn.Sequential(ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, D=self.D),
                          ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D))
        ])

        self.end_mlp = nn.Sequential(
            ME.MinkowskiLinear(8, 16),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(16, 64),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(64, 32),
            ME.MinkowskiReLU()
        )

        self.uam1 = UpSampleAttention(256)
        self.uam1_ = UpSampleAttention(256)
        self.uam2 = UpSampleAttention(128)
        self.uam2_ = UpSampleAttention(128)
        self.uam3 = UpSampleAttention(32)
        self.uam3_ = UpSampleAttention(32)
        self.uam4 = UpSampleAttention(8)
        self.uam4_ = UpSampleAttention(8)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x4 = self.mlp(x4)

        y1 = self.up1[0](x4)  # 256
        u1 = self.uam1(y1, x3)
        u1_ = self.uam1_(x3, x3)
        y1 = ME.cat(u1, u1_)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        u2 = self.uam2(y2, x2)
        u2_ = self.uam2_(x2, x2)
        y2 = ME.cat(u2, u2_)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        u3 = self.uam3(y3, x1)
        u3_ = self.uam3_(x1, x1)
        y3 = ME.cat(u3, u3_)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        u4 = self.uam4(y4, x0)
        u4_ = self.uam4_(x0, x0)
        y4 = ME.cat(u4, u4_)
        y4 = self.up4[1](y4)

        y4 = self.end_mlp(y4)

        return y4


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.projection_head = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.dropout = ME.MinkowskiDropout(p=0.4)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x):
        x = self.dropout(x)  # from input points dropout some (increase randomness)
        x = self.glob_pool(x)  # global max pooling over the remaining points

        out = self.projection_head(x.F)  # project the max pooled features

        return out


class SegmentationClassifierHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=26):
        nn.Module.__init__(self)

        self.dp = ME.MinkowskiDropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        x = self.dp(x)

        return self.fc(x.F)
