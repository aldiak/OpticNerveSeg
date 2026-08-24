import torch
import torch.nn as nn
#from metrics_3d import init_weights, count_param
from torchsummary import summary
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F
### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
class unet3dConv3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, dropout_p, n=2, ks=3, stride=1, padding=1):
        super(unet3dConv3d, self).__init__()
        self.n = n # 第n层
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride=stride, padding=padding),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(dropout_p))
                setattr(self, 'conv%d' % i, conv) # 如果属性不存在会创建一个新的对象属性，并对属性赋值
                                                    # setattr(object, name, value)
                                                    # object - - 对象。
                                                    # name - - 字符串，对象属性。
                                                    # value - - 属性值。
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride=stride, padding=padding),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i) # getattr函数用于返回一个对象属性值。
            x = conv(x)
        return x
class unet3dUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unet3dUp, self).__init__()
        self.conv = unet3dConv3d(in_size , out_size, is_batchnorm=True, dropout_p=0.0)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear3d(scale_factor=2),
                nn.Conv3d(in_size, out_size, 1))
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unet3dConv3d') != -1: continue
            init_weights(m, init_type='kaiming')
    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)
class UNet3d(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, is_deconv=True, is_batchnorm=True):
        super(UNet3d, self).__init__()
        self.in_channels = in_channels # 通道数
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters_base = 64
        # downsampling
        self.maxpool = nn.MaxPool3d(2)
        self.conv1 = unet3dConv3d(self.in_channels, filters_base, self.is_batchnorm, dropout_p=0.1) # 1 64
        self.conv2 = unet3dConv3d(filters_base, filters_base*2, self.is_batchnorm, dropout_p=0.2) # 64 128
        self.conv3 = unet3dConv3d(filters_base*2, filters_base*4, self.is_batchnorm, dropout_p=0.3) # 128 256
        self.conv4 = unet3dConv3d(filters_base*4, filters_base*8, self.is_batchnorm, dropout_p=0.4) # 256 512
        self.center = unet3dConv3d(filters_base*8, filters_base*16, self.is_batchnorm, dropout_p=0.5) # 512 1024
         # f1 and g1 encoder 1280, 1296, 3136, 3600
        self.f1 = nn.Sequential(nn.Conv3d(filters_base*16, 64, kernel_size=3, padding=1, bias=True),
                                nn.Conv3d(64, 16, kernel_size=1))
        self.g1 = nn.Sequential(nn.Linear(in_features=1280, out_features=512),
                                nn.ReLU(),
                                nn.Linear(in_features=512, out_features=256))
        # self.g1 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=512, out_features=256))
        # upsampling
        self.up_concat4 = unet3dUp(filters_base*16, filters_base*8, self.is_deconv) # 1024 512
        self.up_concat3 = unet3dUp(filters_base*8, filters_base*4, self.is_deconv) # 512 256
        self.up_concat2 = unet3dUp(filters_base*4, filters_base*2, self.is_deconv) # 256 128
        self.up_concat1 = unet3dUp(filters_base*2, filters_base, self.is_deconv) # 128 64
        # final conv (without any concat)
        self.final = nn.Conv3d(filters_base, n_classes, 1) # 64 2
        for m in self.modules():
            if isinstance(m, nn.Conv3d): # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                                            # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。。
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                                                # filters input input
    def forward(self, inputs): # in->out 1*512*512 1*64*64
        conv1 = self.conv1(inputs) # 1->64 64*512*512 64*64*64 unet3dConv3d-7:[-1, 64, 64, 64]
        maxpool1 = self.maxpool(conv1) # 64*256*256 64*32*32
        conv2 = self.conv2(maxpool1) # 64->128 128*256*256 128*32*32 unet3dConv3d-15:[-1, 128, 32, 32]
        maxpool2 = self.maxpool(conv2) # 128*128*128 128*16*16
        conv3 = self.conv3(maxpool2) # 128->256 256*128*128 256*16*16 unet3dConv3d-23:[-1, 256, 16, 16]
        maxpool3 = self.maxpool(conv3) # 256*64*64 256*8*8
        conv4 = self.conv4(maxpool3) # 256->512 512*64*64 512*8*8 unet3dConv3d-31:[-1, 512, 8, 8]
        maxpool4 = self.maxpool(conv4) # 512*32*32 512*4*4
        center = self.center(maxpool4) # 512->1024 1024*32*32 1024*4*4 unet3dConv3d-39:[-1, 1024, 4, 4]
        high_d = self.f1(center)
        #print(high_d.shape)
        high_d_represent = self.g1(high_d.reshape(high_d.size(0), -1))
        up4 = self.up_concat4(center, conv4) # 1024->512 512*64*64 512*8*8 unet3dUp-46:[-1, 512, 8, 8]
        up3 = self.up_concat3(up4, conv3) # 512->256 256*128*128 256*16*16 unet3dUp-53:[-1, 256, 16, 16]
        up2 = self.up_concat2(up3, conv2) # 256->128 128*256*256 128*32*32 unet3dUp-60:[-1, 128, 32, 32]
        up1 = self.up_concat1(up2, conv1) # 128->64 64*512*512 64*64*64 unet3dUp-67:[-1, 64, 64, 64]
        final1 = self.final(up1)
        final = final1 #nn.Sigmoid()(final1)
        return final, high_d_represent, center  # Added center for spatial feature alignment