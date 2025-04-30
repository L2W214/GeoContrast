import torch.nn as nn
import torch

import MinkowskiEngine as ME


def cam_output(x, att):
    source = att.decomposed_features
    x_f = x.decomposed_features
    result = []

    # ====================================== 优化for循环，提高速度 ======================================
    for i in range(len(x_f)):
        result.append(x_f[i] * source[i] + x_f[i])

    result = torch.vstack(result)
    output = ME.SparseTensor(features=result, coordinate_manager=x.coordinate_manager,
                             coordinate_map_key=x.coordinate_map_key)

    return output


def sam_mid(x, att):
    source = att.decomposed_features
    x_f = x.decomposed_features
    result = []

    # ====================================== 优化for循环，提高速度 ======================================
    for i in range(len(x_f)):
        result.append(x_f[i] * source[i])

    result = torch.vstack(result)
    output = ME.SparseTensor(features=result, coordinate_manager=x.coordinate_manager,
                             coordinate_map_key=x.coordinate_map_key)

    return output


def sam_output(mid_, final, x):
    """
    [(mid_ * final) + mid_] + x
    """
    x_f = x.decomposed_features
    x_f = torch.vstack(x_f)

    mid_ = mid_.decomposed_features
    mid_ = torch.vstack(mid_)

    final = final.decomposed_features
    final = torch.vstack(final)
    final = torch.max(final, dim=1)[0].unsqueeze(-1)

    out_put = torch.mul(mid_, final)
    out_put = (out_put + mid_) + x_f

    out_put = ME.SparseTensor(features=out_put, coordinate_manager=x.coordinate_manager,
                              coordinate_map_key=x.coordinate_map_key)

    return out_put


class ScaleAttentionMechanism(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super().__init__()

        self.avg_pooled = ME.MinkowskiGlobalAvgPooling()
        self.max_pooled = ME.MinkowskiGlobalMaxPooling()

        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(in_channels, in_channels // reduction_ratio),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(in_channels // reduction_ratio, in_channels)
        )

        self.sigmoid_ = ME.MinkowskiSigmoid()

        self.spatial_attention = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(4),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(4, 4, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiSigmoid()
        )

    def forward(self, x):
        avg_x = self.avg_pooled(x)
        max_x = self.max_pooled(x)

        avg_out = self.mlp(avg_x)
        max_out = self.mlp(max_x)
        out = avg_out + max_out

        weights = self.sigmoid_(out)
        output_mid = sam_mid(x, weights)
        output = self.spatial_attention(output_mid)
        output = sam_output(output_mid, output, x)

        return output


class ChannelAttentionModel(nn.Module):
    def __init__(self, in_channel, reduction_ratio=2):
        super(ChannelAttentionModel, self).__init__()

        self.avg_pooled = ME.MinkowskiGlobalAvgPooling()
        self.max_pooled = ME.MinkowskiGlobalMaxPooling()

        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(in_channel, in_channel//reduction_ratio),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(in_channel//reduction_ratio, in_channel)
        )

        self.sigmoid_ = ME.MinkowskiSigmoid()

    def forward(self, x):
        avg_x = self.avg_pooled(x)
        max_x = self.max_pooled(x)

        avg_out = self.mlp(avg_x)
        max_out = self.mlp(max_x)
        out = avg_out + max_out

        attention_weights = self.sigmoid_(out)
        output = cam_output(x, attention_weights)

        return output


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

        self.cam1 = ChannelAttentionModel(cs[5])
        self.cam2 = ChannelAttentionModel(cs[6])
        self.cam3 = ChannelAttentionModel(cs[7])
        self.cam4 = ChannelAttentionModel(cs[8])

        self.sam = ScaleAttentionMechanism(16)

        self.up_sample1 = BasicDeconvolutionBlock(cs[5], cs[5], ks=8, stride=8, D=self.D)
        self.up_sample2 = BasicDeconvolutionBlock(cs[6], cs[6], ks=4, stride=4, D=self.D)
        self.up_sample3 = BasicDeconvolutionBlock(cs[7], cs[7], ks=2, stride=2, D=self.D)

        self.squeeze1 = ME.MinkowskiConvolution(cs[5], 4, kernel_size=1, stride=1, dimension=self.D)
        self.squeeze2 = ME.MinkowskiConvolution(cs[6], 4, kernel_size=1, stride=1, dimension=self.D)
        self.squeeze3 = ME.MinkowskiConvolution(cs[7], 4, kernel_size=1, stride=1, dimension=self.D)
        self.squeeze4 = ME.MinkowskiConvolution(cs[8], 4, kernel_size=1, stride=1, dimension=self.D)

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
            ME.MinkowskiLinear(16, 64),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(64, 32),
            ME.MinkowskiReLU()
        )

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
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)
        y1 = self.cam1(y1)  # 256

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)
        y2 = self.cam2(y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)
        y3 = self.cam3(y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)
        y4 = self.cam4(y4)  # 8

        if self.is_cat:
            y1 = self.squeeze1(self.up_sample1(y1))
            y2 = self.squeeze2(self.up_sample2(y2))
            y3 = self.squeeze3(self.up_sample3(y3))
            y4 = self.squeeze4(y4)

            out = ME.cat(y1, y2, y3, y4)
            out = self.sam(out)
            out = self.end_mlp(out)

            return out
        else:
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
        x = self.dropout(x)    # from input points dropout some (increase randomness)
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
