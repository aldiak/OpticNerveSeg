import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Network components from utils.py and others
def init_weights(net, init_type="normal"):
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    # Add other init types if needed


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UnetConv3(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        is_batchnorm,
        kernel_size=(3, 3, 3),
        padding_size=(1, 1, 1),
        init_stride=(1, 1, 1),
    ):
        super(UnetConv3, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3_CT(nn.Module):  # Assuming similar to UnetUp3
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(
            out_size * 2,
            out_size,
            is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.up = nn.ConvTranspose3d(
            in_size, out_size, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = (
            outputs2.size()[2] - inputs1.size()[2],
            outputs2.size()[3] - inputs1.size()[3],
            outputs2.size()[4] - inputs1.size()[4],
        )
        padding = [
            offset[2] // 2,
            offset[2] // 2,
            offset[1] // 2,
            offset[1] // 2,
            offset[0] // 2,
            offset[0] // 2,
        ]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


# unet_3D from unet_3D.py
class unet_3D(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=2,
        is_batchnorm=True,
    ):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(
            self.in_channels,
            filters[0],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(
            filters[0],
            filters[1],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(
            filters[1],
            filters[2],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(
            filters[2],
            filters[3],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(
            filters[3],
            filters[4],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final


class unet_3D_dt(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=2,
        is_deconv=True,
        in_channels=2,
        is_batchnorm=True,
    ):
        super(unet_3D_dt, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(
            self.in_channels,
            filters[0],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(
            filters[0],
            filters[1],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(
            filters[1],
            filters[2],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(
            filters[2],
            filters[3],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(
            filters[3],
            filters[4],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.tanh = nn.Tanh()

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)
        dis = self.tanh(final)

        return dis, final


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode="trilinear"),
        )

    def forward(self, input):
        return self.dsv(input)


class unet_3D_dv_semi(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=21,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet_3D_dv_semi, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(
            self.in_channels,
            filters[0],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(
            filters[0],
            filters[1],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(
            filters[1],
            filters[2],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(
            filters[2],
            filters[3],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(
            filters[3],
            filters[4],
            self.is_batchnorm,
            kernel_size=(3, 3, 3),
            padding_size=(1, 1, 1),
        )

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(
            in_channels=filters[0], out_channels=n_classes, kernel_size=1
        )

        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up4 = self.dropout1(up4)

        up3 = self.up_concat3(conv3, up4)
        up3 = self.dropout2(up3)

        up2 = self.up_concat2(conv2, up3)
        up2 = self.dropout3(up2)

        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout4(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        return dsv1, dsv2, dsv3, dsv4

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
