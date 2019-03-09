import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import numpy as np
import math
import sys
import os
from generateFeature import GFM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'handmodelLayer'))
from resnet import resnet18
from resnet import resnet50
from resnet import  resnet34
# from resnet import seresnet18
# from resnet import seresnet50
from resnet import FPN
# from resnet import *
from resnet import Bottleneck
from resnet import BasicBlock

"""
res module
"""
class res2d_(nn.Module):
    expansion = 1

    def __init__(self, inchannels, outchannels, stride=1, option=False):
        super(res2d_, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.conv_res = nn.Conv2d(inchannels, outchannels,kernel_size=1, stride=1, padding=0)
        self.bn_res = nn.BatchNorm2d(outchannels)
        self.stride = stride
        self.inchannels = inchannels
        self.outchannels = outchannels

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.inchannels != self.outchannels:
            residual = self.conv_res(residual)
            residual = self.bn_res(residual)

        out += residual
        out = self.relu(out)
        return out

class RegionEnsemble(nn.Module):

    def __init__(self, feat_size=12):
        assert((feat_size/4).is_integer())
        super(RegionEnsemble, self).__init__()
        self.feat_size = feat_size
        self.grids = nn.ModuleList()
        for i in range(9):
            self.grids.append(self.make_block(self.feat_size))

    def make_block(self, feat_size):
        size = int(self.feat_size/2)
        return nn.Sequential(nn.Linear(64*size*size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout())

    def forward(self, x):

        midpoint = int(self.feat_size/2)
        quarterpoint1 = int(midpoint/2)
        quarterpoint2 = int(quarterpoint1 + midpoint)
        regions = []
        ensemble = []

        #4 corners
        regions += [x[:, :, :midpoint, :midpoint].clone(), x[:, :, :midpoint, midpoint:].clone(), x[:, :, midpoint:, :midpoint].clone(), x[:, :, midpoint:, midpoint:].clone()]
        # 4 overlapping centers

        regions += [x[:, :, quarterpoint1:quarterpoint2, :midpoint].clone(), x[:, :, quarterpoint1:quarterpoint2, midpoint:].clone(), x[:, :, :midpoint, quarterpoint1:quarterpoint2].clone(), x[:, :, midpoint:, quarterpoint1:quarterpoint2].clone()]
        # middle center
        regions += [x[:, :, quarterpoint1:quarterpoint2, quarterpoint1:quarterpoint2].clone()]

        for i in range(0,9):
            out = regions[i]
            # print(out.shape)
            out = out.contiguous()
            out = out.view(out.size(0),-1)
            out = self.grids[i](out)
            ensemble.append(out)

        out = torch.cat(ensemble,1)

        return out

class Residual(nn.Module):

    def __init__(self, planes):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += residual
        return out

class REN(nn.Module):

    def __init__(self, inputSz, outputdim):
        super(REN, self).__init__()
        feat = np.floor(((inputSz - 1 -1)/2) +1)
        feat = np.floor(((feat - 1-1)/2) +1)
        feat = np.floor(((feat - 1-1)/2) +1)
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.bn = nn.InstanceNorm2d(1)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2_dim_inc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 1, padding=0)
        self.relu2 = nn.ReLU()
        self.res1 = Residual(planes = 32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.conv3_dim_inc = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 1, padding=0)
        self.relu4 = nn.ReLU()
        self.res2 = Residual(planes = 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.region_ens = RegionEnsemble(feat_size=feat)
        #class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(9*2048, outputdim)

    def forward(self, x):

        # out = self.bn(x)
        out = self.conv0(x)
        out = self.relu0(out)

        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.relu1(out)

        out = self.conv2_dim_inc(out)
        out = self.relu2(out)

        out = self.res1(out)

        out = self.maxpool2(out)
        out = self.relu3(out)

        out = self.conv3_dim_inc(out)
        out = self.relu4(out)

        out = self.res2(out)

        out = self.maxpool3(out)
        out = self.relu5(out)        #relu5
        out = self.dropout(out)


        #slice
        out = self.region_ens(out)
        # flatten the output
        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        return out, None
"""
joint module in stage
"""
class JM_S_(nn.Module):
    expansion = 1

    def __init__(self, input_size, output_dim, stride=1, inchannels=1, dim=16):
        super(JM_S_, self).__init__()
        self.feature_size = input_size/2
        self.conv0 = nn.Conv2d(inchannels, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1)
        self.res1 = res2d_(dim*2, dim*2, stride)
        self.conv3 = nn.Conv2d(dim*2, dim * 4, kernel_size=3, padding=1)
        self.res2 = res2d_(dim * 4, dim * 4, stride)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.feature_size * self.feature_size * dim * 4, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024,output_dim)
        self.output_dim = output_dim
        self.stride = stride
        self.dim = dim

    def forward(self, x):

        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.res1(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.res2(out)
        out = self.pool3(out)
        out = self.drop(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = out.view(-1, self.output_dim)
        return out

    def num_flat_features(selfself, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# JRM head
class JM_H_(nn.Module):
    expansion = 1

    def __init__(self, input_size, output_dim, stride=1, inchannels=1, dim=16):
        super(JM_H_, self).__init__()
        self.feature_size = input_size / 8
        self.conv0 = nn.Conv2d(inchannels, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1)
        self.res1 = res2d_(dim*2, dim*2, stride)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(dim*2, dim * 4, kernel_size=3, padding=1)
        self.res2 = res2d_(dim * 4, dim * 4, stride)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.feature_size*self.feature_size*dim*4, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_dim)
        self.stride = stride
        self.dim = dim
        self.output_dim = output_dim

    def forward(self, x):

        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.pool1(out)
        pool1 = self.relu(out)
        out = self.conv2(pool1)
        out = self.relu(out)

        out = self.res1(out)

        out = self.pool2(out)
        pool2 = self.relu(out)
        out = self.conv3(pool2)
        out = self.relu(out)

        out = self.res2(out)

        out = self.pool2(out)
        out = self.relu(out)
        out = self.drop(out)
        # print(out.size())
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = out.view(-1, self.output_dim)
        return pool1, pool2, out

    def num_flat_features(selfself, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#head
class HD_(nn.Module):
    expansion = 1

    def __init__(self, inchannels,out_channels,option=False):
        super(HD_, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, out_channels/2, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = res2d_(out_channels/2, out_channels)
        self.res2 = res2d_(out_channels,  out_channels)
        self.res3 = res2d_(out_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out


#hourglass modules
class HG_(nn.Module):
    expansion = 1

    def __init__(self, inchannels, J):
        super(HG_, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1_e = res2d_(inchannels, 2*inchannels)
        self.res2_e = res2d_(2*inchannels, 4*inchannels)
        self.res3_e = res2d_(4*inchannels, 8*inchannels)
        self.res1_d = res2d_(2*inchannels, inchannels)
        self.res2_d = res2d_(4*inchannels, 2*inchannels)
        self.res3_d = res2d_(8*inchannels, 4*inchannels)
        self.uppool1 = nn.ConvTranspose2d(inchannels, inchannels, kernel_size=2,stride=2)
        self.uppool2 = nn.ConvTranspose2d(2*inchannels, 2*inchannels, kernel_size=2,stride=2)
        self.uppool3 = nn.ConvTranspose2d(4*inchannels, 4*inchannels, kernel_size=2,stride=2)
        self.res3_c = res2d_(8*inchannels, 8*inchannels)
        self.res2_c = res2d_(4*inchannels, 4*inchannels)
        self.res1_c = res2d_(2*inchannels, 2*inchannels)
        self.res0_c = res2d_(inchannels, inchannels)


        self.res = res2d_(inchannels, inchannels)
        self.con1 = nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1, padding=0)
        self.con2 = nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1, padding=0)
        self.heat = nn.Conv2d(inchannels, J, kernel_size=1, stride=1, padding=0)
        self.conheat = nn.Conv2d(J, inchannels, kernel_size=1, stride=1, padding=0)
        # self.cov1 = nn.Conv2d(inchannels/2, inchannels/4, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        ## x:64 res1:32 res2:16 res3:8
        res1 = self.pool1(x)
        res0_c = self.res0_c(x)
        res1 = self.res1_e(res1)


        res2 = self.pool2(res1)
        res1_c = self.res1_c(res1)
        res2 = self.res2_e(res2)

        res3 = self.pool3(res2)
        res2_c = self.res2_c(res2)

        res3 = self.res3_e(res3)
        res3 = self.res3_c(res3)

        res3_out = self.res3_d(res3)
        res3_out = self.uppool3(res3_out)

        res3_out +=res2_c

        res2_out = self.res2_d(res3_out)
        res2_out = self.uppool2(res2_out)
        res2_out += res1_c

        res1_out = self.res1_d(res2_out)
        res1_out = self.uppool1(res1_out)
        res1_out += res0_c

        y = self.res(res1_out)
        y = self.con1(y)
        heatmap = self.heat(y)
        heat_ = self.conheat(heatmap)
        y_ = self.con2(y)
        out = x + y_ + heat_
        return out, heatmap


    def num_flat_features(selfself, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# MD_:feature map block
class FMB_(nn.Module):
    def __init__(self, inchannels):
        super(FMB_, self).__init__()
        self.uppool0 = nn.ConvTranspose2d(inchannels, inchannels/2, kernel_size=2, stride=2)
        self.relu0 = nn.ReLU()
        self.uppool1 = nn.ConvTranspose2d(inchannels/2, inchannels/4, kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.uppool0(x)
        out = self.relu0(out)
        out = self.uppool1(out)
        out = self.relu1(out)
        return out


# MD_:linea
# r map block
class LMB_(nn.Module):
    def __init__(self, J, outchannels, outsize):
        super(LMB_, self).__init__()
        self.fc1 = nn.Linear(J*3, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, outchannels*outsize*outsize)
        self.outchannels = outchannels
        self.outsize = outsize
        self.J = J

    def forward(self, x):
        x = x.view(-1, self.J*3)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, self.outchannels, self.outsize, self.outsize)
        return out


# MD_:linear relu map module
class LRMB_(nn.Module):
    def __init__(self, J, outchannels, outsize):
        super(LRMB_, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(J * 3, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, outchannels * outsize * outsize)
        self.outchannels = outchannels
        self.outsize = outsize
        self.J = J

    def forward(self, x):
        x = x.view(-1, self.J*3)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = out.view(-1, self.outchannels, self.outsize, self.outsize)
        return out


class MyReLU(torch.autograd.Function):

    def forward(self, input_):

        self.save_for_backward(input_)
        output = input_.clamp(min=0)
        return output

    def backward(self, grad_output):

        input_, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class muti_net_HG(nn.Module):
    def __init__(self, J,hg_channel):
        super(muti_net_HG, self).__init__()
        self.process = HD_(1, hg_channel)#in_channels,out_channels 48*48*16
        self.stage0 = HG_(hg_channel,J,48)
        self.map0 = LMB_(J, 1, 48)#outchannels, outsize
        #self.cov0 = nn.Conv2d(hg_channel+1, hg_channel, kernel_size=1, stride=1, padding=0)
        self.stage1 = HG_(hg_channel+1,J,48)
        self.map1 = LMB_(J, 1, 48)
        #self.cov1 = nn.Conv2d(hg_channel+1, hg_channel, kernel_size=1, stride=1, padding=0)
        self.stage2 = HG_(hg_channel+1,J,48)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.001)
                # m.weight.data.normal_(0, 0.001)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x):
        process= self.process(x)
        s0 =self.stage0(process)
        s = self.map0(s0)

        s = torch.cat((process, s), 1)
        s1 =self.stage1(s)
        s = self.map1(s1)

        s = torch.cat((process, s), 1)
        s2 = self.stage2(s)
        return s0, s1, s2


class Modified_SmoothL1Loss(torch.nn.Module):

    def __init__(self):
        super(Modified_SmoothL1Loss,self).__init__()

    def forward(self,x,y):
        total_loss = 0
        assert(x.shape == y.shape)
        z = (x - y).float()
        mse = (torch.abs(z) < 0.01).float() * z
        l1 = (torch.abs(z) >= 0.01).float() * z
        total_loss += torch.sum(self._calculate_MSE(mse))
        total_loss += torch.sum(self._calculate_L1(l1))

        return total_loss/z.shape[0]

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)

class HMM(nn.Module):
    def __init__(self, args):
        super(HMM, self).__init__()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        self.stage0 = JM_H_(args.inputSz, args.model_para)

    def forward(self, x):
        poo1, pool2, theta = self.stage0(x)
        pos = self.handmodelLayer.calculate_position(theta)
        return pos


class JRM(nn.Module):
    def __init__(self,args):
        super(JRM, self).__init__()
        self.stage0 = JM_H_(args.inputSz, args.model_para)

    def forward(self, x):
        poo1, pool2, pos = self.stage0(x)
        return pos


#B*1*128*128
class HG_res(nn.Module):
    expansion = 1

    def __init__(self, hg_channels, J):
        super(HG_res, self).__init__()
        self.hg_channels = hg_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(1, hg_channels/2, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = res2d_(hg_channels/2, hg_channels)
        self.res2 = res2d_(hg_channels,  hg_channels)
        self.hg1 = HG_(hg_channels, J)
        self.hg2 = HG_(hg_channels, J)

        image_factor = 4
        channel_factor = 1
        self.res_reg1 = res2d_(hg_channels, hg_channels)
        self.res_reg2 = res2d_(hg_channels,  hg_channels)
        self.pool_reg1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_reg3 = res2d_(hg_channels, hg_channels)
        self.pool_reg2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_reg4 = res2d_(hg_channels,  hg_channels)
        self.fc1 = nn.Linear(hg_channels/channel_factor*64/image_factor*64/image_factor, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, J)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.res1(out)
        out = self.res2(out)
        out, heatmap1 = self.hg1(out)
        out, heatmap2 = self.hg2(out)

        out = self.res_reg1(out)
        out = self.res_reg2(out)
        out = self.pool_reg1(out)
        out = self.res_reg3(out)
        out = self.pool_reg2(out)
        out = self.res_reg4(out)
        out = out.view(out.size()[0], -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return heatmap1, heatmap2, out


# class muti_net(nn.Module):
#     def __init__(self, args):
#         super(muti_net, self).__init__()
#         if args.dataset =='nyu':
#             self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
#         elif args.dataset =='icvl':
#             self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
#         elif args.dataset =='msra':
#             self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
#
#
#         J = args.J
#         self.stage0 = JM_H_(args.inputSz, J*3, inchannels=1)
#         self.map1 = LMB_(J, 4, args.inputSz/4)
#         self.stage1 = JM_S_(args.inputSz/4, args.model_para, inchannels=36)
#         self.map2 = LMB_(self.handmodelLayer.J, 4, args.inputSz/4)
#         self.stage2 = JM_S_(args.inputSz/4, J*3, inchannels=36)
#         # self.map3 = LMB_(J, 4, args.inputSz/4)
#         # self.stage3 = JM_S_(args.inputSz/4, args.model_para, inchannels=36)
#         # self.map4 = LMB_(self.handmodelLayer.J, 4, args.inputSz/4)
#         # self.stage4 = JM_S_(args.inputSz/4, J*3, inchannels=36)
#
#     def forward(self, x):
#         poo1, pool2, pos0 = self.stage0(x)
#         feat1 = self.map1(pos0)
#         theta = self.stage1(torch.cat((pool2,feat1),1))
#         pos1 = self.handmodelLayer.calculate_position(theta)
#         feat2 = self.map2(pos1)
#         pos2 = self.stage2(torch.cat((pool2,feat2),1))
#         return pos0, pos1, pos2,feat1, feat2
#         # feat3 = self.map1(pos2)
#         # theta = self.stage1(torch.cat((pool2,feat3),1))
#         # pos3 = self.handmodelLayer.calculate_position(theta)
#         # feat4 = self.map2(pos3)
#         # pos4 = self.stage2(torch.cat((pool2,feat4),1))
#         # return pos0,pos1,pos2,pos3,pos4,feat1,feat2,feat3,feat4
#

class muti_net_feat(nn.Module):
    def __init__(self, args):
        super(muti_net_feat, self).__init__()
        J = args.J
        self.stage0 = JM_H_(1, 16,output_dim=J*3)
        self.map1 = LMB_(J, 4, 24)
        self.stage1 = JM_S_(36, 16,output_dim=J*3)
        self.map2 = LMB_(J, 4, 24)
        self.stage2 = JM_S_(36, 16,output_dim=J*3)

    def forward(self, x):
        poo1, pool2, pos0 = self.stage0(x)
        feat1 = self.map1(pos0)
        pos1 = self.stage1(torch.cat((pool2,feat1),1))
        feat2 = self.map2(pos1)
        pos2 = self.stage2(torch.cat((pool2,feat2),1))
        return pos0,pos1,pos2,feat1,feat2


class muti_net_heatmap(nn.Module):
    def __init__(self, args):
        super(muti_net_heatmap, self).__init__()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        self.util1 = GFM(args.batchSz, args.gpu_id, args.inputSz / 2)
        self.util2 = GFM(args.batchSz, args.gpu_id, args.inputSz / 2)
        self.J = args.J
        self.inputSz = args.inputSz
        self.stage0 = JM_H_(args.inputSz, self.J*3, inchannels=1)
        self.stage1 = JM_S_(args.inputSz/2, args.model_para, inchannels=(self.J+16))
        self.stage2 = JM_S_(args.inputSz/2, self.J*3, inchannels=(self.handmodelLayer.J+16))


    def forward(self, x):
        poo1, pool2, pos0 = self.stage0(x)
        feat1 = self.util1.joint2heatmap2d(pos0.view(-1, self.J, 3))
        theta = self.stage1(torch.cat((poo1, feat1.detach()), 1))
        # theta = self.stage1(torch.cat((poo1, feat1), 1))
        pos1 = self.handmodelLayer.calculate_position(theta)
        feat2 = self.util2.joint2heatmap2d(pos1.view(-1, self.handmodelLayer.J, 3))
        pos2 = self.stage2(torch.cat((poo1,feat2.detach()),1))
        # pos2 = self.stage2(torch.cat((poo1,feat2),1))
        return pos0, pos1, pos2, feat1, feat2


class muti_net_noconnect(nn.Module):
    def __init__(self, args):
        super(muti_net_noconnect, self).__init__()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        self.J = args.J
        self.inputSz = args.inputSz
        self.stage0 = JM_H_(args.inputSz, self.J*3, inchannels=1)
        self.stage1 = JM_S_(args.inputSz/4, args.model_para, inchannels=32)
        self.stage2 = JM_S_(args.inputSz/4, self.J*3, inchannels=32 )


    def forward(self, x):
        poo1, pool2, pos0 = self.stage0(x)
        theta = self.stage1(pool2)
        pos1 = self.handmodelLayer.calculate_position(theta)
        pos2 = self.stage2(pool2)
        return pos0, pos1, pos2, poo1, pool2


#define for fpn
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
    def forward(self, up1, up2, up3, up4):
        return torch.cat((up1, up2, up3, up4), 1)


class FPNnet(nn.Module):
    def __init__(self, args):
        super(FPNnet, self).__init__()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        self.J = args.J
        self.inputSz = args.inputSz

        # resnet50
        # self.fpn = FPN(Bottleneck, [3, 4, 6, 3])
        # max_planes = 2048
        # min_planes = 256
        # resnet18
        self.fpn = FPN(BasicBlock, [2, 2, 2, 2])
        max_planes = 512
        min_planes = 64
        ##################################################################################
        # hand model subnet
        # self.convh1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.pool_model = nn.AvgPool2d(8, stride=1)
        self.fc_model = nn.Linear(max_planes, args.model_para)

        ##################################################################################
        # heatmap subnet
        # intermediate supervision
        self.convfin_k2 = nn.Conv2d(min_planes, args.J, kernel_size=1, stride=1, padding=0)
        self.convfin_k3 = nn.Conv2d(min_planes, args.J, kernel_size=1, stride=1, padding=0)
        self.convfin_k4 = nn.Conv2d(min_planes, args.J, kernel_size=1, stride=1, padding=0)
        self.convfin_k5 = nn.Conv2d(min_planes, args.J, kernel_size=1, stride=1, padding=0)
        # 2 conv(kernel=3x3),change channels from 256 to 128
        self.convt1 = nn.Conv2d(min_planes, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.Conv2d(min_planes, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.Conv2d(min_planes, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.Conv2d(min_planes, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.Conv2d(min_planes / 2, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convs2 = nn.Conv2d(min_planes / 2, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convs3 = nn.Conv2d(min_planes / 2, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.convs4 = nn.Conv2d(min_planes / 2, min_planes / 2, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest', align_corners=None)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest', align_corners=None)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.concat = Concat()
        self.conv2 = nn.Conv2d(min_planes / 2 * 4, min_planes, kernel_size=3, stride=1, padding=1)
        self.convfin = nn.Conv2d(min_planes, args.J, kernel_size=1, stride=1, padding=0)


    def keypoint_forward(self,features):
        saved_for_loss = []
        p2, p3, p4, p5 = features # fpn features for keypoint subnet
        ##################################################################################
        # keypoints subnet
        # intermediate supervision
        saved_for_loss.append(self.convfin_k2(p2))
        saved_for_loss.append(self.upsample3(self.convfin_k3(p3)))
        saved_for_loss.append(self.upsample2(self.convfin_k4(p4)))
        saved_for_loss.append(self.upsample1(self.convfin_k5(p5)))
        #
        p5 = self.convt1(p5)
        p5 = self.convs1(p5)
        p4 = self.convt2(p4)
        p4 = self.convs2(p4)
        p3 = self.convt3(p3)
        p3 = self.convs3(p3)
        p2 = self.convt4(p2)
        p2 = self.convs4(p2)
        p5 = self.upsample1(p5)
        p4 = self.upsample2(p4)
        p3 = self.upsample3(p3)
        predict_keypoint = self.convfin(F.relu(self.conv2(self.concat(p5, p4, p3, p2))))
        saved_for_loss.append(predict_keypoint)
        return predict_keypoint, saved_for_loss

    def forward(self, x):
        features = self.fpn(x)
        _, heatmap = self.keypoint_forward(features[1])
        c2, c3, c4, c5 = features[0]  # fpn features for handmodel subnet
        para = self.pool_model(c5)
        para = para.view(para.size(0),-1)
        para = self.fc_model(para)
        joint = self.handmodelLayer.calculate_position(para)
        return heatmap, joint



class MREN(nn.Module):
    def __init__(self, args):
        super(MREN, self).__init__()
        self.ren = REN(args.inputSz,args.model_para)
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        self.J = args.J

    def forward(self, x):
        out = self.ren(x)
        out = self.handmodelLayer.calculate_position(out).view(-1, self.J, 3)
        return out


class Mresnet(nn.Module):
    def __init__(self, args):
        super(Mresnet, self).__init__()
        model_name = args.model_type[2:]
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = resnet18(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 1024)
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = resnet50(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 1024)
        elif model_name == "seresnet18":
            """ SEResnet18
            """
            model_ft = seresnet18(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 1024)
        elif model_name == "seresnet50":
            """ SEResnet50
            """
            model_ft = seresnet18(pretrained=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 1024)
        else:
            print("Invalid model name, exiting...")
            exit()

        self.res = model_ft
        self.fc_theta = nn.Linear(1024, args.model_para)
        # self.fc_theta = nn.Linear(1024, 36)
        # self.fc_bonelen = nn.Linear(1024, 15)
        self.sm = nn.Sigmoid()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id,mesh_dir='/data/users/pfren/pycharm/deeppiror/src/handmodelLayer/hand_mesh', mesh_type='wrist')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        self.J = args.J

    def forward(self, x):
        out, features = self.res(x)
        theta = self.fc_theta(out)
        theta[:, 0:3] = self.sm(theta[:, 0:3]) / 2.0
        # bone = self.sm(self.fc_bonelen(out))
        # bone = torch.sin(self.fc_bonelen(out))
        # out = torch.cat((theta, bone), dim=1)
        out = self.handmodelLayer.calculate_position(theta)
        return out, theta


class muti_net_res(nn.Module):
    def __init__(self, args):
        super(muti_net_res, self).__init__()
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id, mesh_type='dense')


        J = args.J
        self.GFM_ = GFM(args.batchSz, args.gpu_id, args.inputSz / 2)
        self.map1 = LMB_(J, 4, args.inputSz / 8)
        self.map2 = LMB_(J, 4, args.inputSz / 8)

        self.base = resnet18(pretrained=False)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, J*3)
        # c3
        self.res_handmodel = nn.Sequential(_make_layer(BasicBlock, 256, 256, 2),
                                           _make_layer(BasicBlock, 256, 512, 2, stride=2),
                                           nn.AvgPool2d(4, stride=1)
                                           )
        self.fc_handmodel = nn.Linear(512, args.model_para)
        self.res_refine = nn.Sequential(_make_layer(BasicBlock, 256, 256, 2),
                                        _make_layer(BasicBlock, 256, 512, 2, stride=2),
                                        nn.AvgPool2d(4, stride=1)
                                        )


        self.mask = maskHead(256)
        self.fc_refine = nn.Linear(512, J * 3)

        # self.GD = nn.Sequential(self._make_layer(BasicBlock, 6, 64, 2),
        #                         self._make_layer(BasicBlock, 64, 128, 2),
        #                         nn.AvgPool2d(31, stride=1))
        # self.fc_refine = nn.Linear(512 + 128, J * 3)




    def forward(self, x):
        pos1,c1,c2,c3,c4 = self.base(x)

        out = self.res_handmodel(c3)
        out = out.view(out.size(0), -1)

        theta = self.fc_handmodel(out)
        pos2 = self.handmodelLayer.calculate_position(theta)
        # heatmap = self.util1.joint2heatmap2d(pos2)

        mask =self.GFM_.img2mask(x)
        mask_c3 = self.mask(c3,mask)
        out = self.res_refine(mask_c3)
        out = out.view(out.size(0), -1)
        # x_GD = self.util1.joint2GD(pos2)
        # x_GD = self.GD(x_GD)

        # x_GD = x_GD.view(x_GD.size(0), -1)
        # x = torch.cat((x, x_GD),1)
        pos3 = self.fc_refine(out)
        return pos1, pos2, pos3, c3, c4


class muti_net(nn.Module):
    def __init__(self, args):
        super(muti_net, self).__init__()
        # feature generate
        self.GFM_ = GFM(args.batchSz, args.gpu_id, args.inputSz / 2, args.J)
        # hand model
        if args.dataset =='nyu':
            self.handmodelLayer = NYUHandmodel(args.batchSz, args.gpu_id, mesh_dir='/data/users/pfren/pycharm/deeppiror/src/handmodelLayer/hand_mesh',mesh_type='dense')
        elif args.dataset =='icvl':
            self.handmodelLayer = ICVLHandmodel(args.batchSz, args.gpu_id)
        elif args.dataset =='msra':
            self.handmodelLayer = MSRAHandmodel(args.batchSz, args.gpu_id)
        # connect method
        if "remap" in args.model_type:
            self.head = remapHead(256, 512, args.J, 8)
            self.pool = nn.AvgPool2d(4, stride=1)
        elif "mesh" in args.model_type:
            self.head = meshHead(256, args.J)
        elif "no_connect" in args.model_type:
            self.head = nn.Sequential(_make_layer(BasicBlock, 256, 256, 2),
                                               _make_layer(BasicBlock, 256, 512, 2, stride=2),
                                               nn.AvgPool2d(4, stride=1)
                                               )
            self.head_pool = nn.Linear(512, args.J*3)
        self.base = resnet18(pretrained=False)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, args.J*3)

        # input is c3
        self.res_handmodel = nn.Sequential(_make_layer(BasicBlock, 256, 256, 2),
                                           _make_layer(BasicBlock, 256, 512, 2, stride=2),
                                           nn.AvgPool2d(4, stride=1)
                                           )
        self.fc_handmodel = nn.Linear(512, args.model_para)



    def forward(self, x):
        pos0, feature = self.base(x)

        c2, c3, c4, c5 = feature
        out = self.res_handmodel(c4)
        out = out.view(out.size(0), -1)
        theta = self.fc_handmodel(out)
        pos1 = self.handmodelLayer.calculate_position(theta)
        # mesh = self.handmodelLayer.calculate_mesh()
        # mesh_img = self.handmodelLayer.mesh2pcl(mesh, 128)
        # pos2 = self.head(mesh_img)
        out = self.head(c4)
        out = out.view(out.size(0), -1)
        pos2 = self.head_pool(out)
        return pos0, pos1, pos2, []
        # return pos0, pos1, pos2, mesh_img


class resnet_head(nn.Module):
    def __init__(self, args):
        super(resnet_head, self).__init__()
        self.GFM_ = GFM(args.batch_size, args.gpu_id, args.input_size / 2, args.joint_num)

        self.res = resnet18(pretrained=False)
        num_ftrs = self.res.fc.in_features
        self.res.fc = nn.Linear(num_ftrs, 1024)

        if "DA" in args.G_type:
            self.head = DAHead(256, 256)
            self.pool = nn.AvgPool2d(8, stride=1)
        elif "heatmap" in args.G_type:
            self.head = heatmapHead(256+args.joint_num, 512)
            self.pool = nn.AvgPool2d(4, stride=1)
        elif "edtmap" in args.G_type:
            self.head = edtmapHead(64+args.joint_num, 512, self.GFM_)
            self.pool = nn.AvgPool2d(4, stride=1)
        elif "noconnect" in args.G_type:
            self.head = _make_layer(BasicBlock, 256, 512, 2, stride=2)
            self.pool = nn.AvgPool2d(4, stride=1)
        elif "remap" in args.G_type:
            self.head = remapHead(256,512,args.joint_num,8)
            self.pool = nn.AvgPool2d(4, stride=1)
        elif "muti_super" in args.G_type:
            self.pool1 = nn.AvgPool2d(32, stride=1)
            self.pool2 = nn.AvgPool2d(16, stride=1)
            self.pool3 = nn.AvgPool2d(8, stride=1)
            self.pool4 = nn.AvgPool2d(4, stride=1)
        elif "mesh" in args.G_type:
            self.head = meshHead(256, args.joint_num)
            self.pool = nn.AvgPool2d(8, stride=1)
        elif "vote" in args.G_type:
            self.head = voteHead(256, args.joint_num)
        elif "model" in args.G_type:
            if args.dataset =='nyu':
                self.handmodelLayer = NYUHandmodel(args.batch_size, args.gpu_id)
            elif args.dataset =='icvl':
                self.handmodelLayer = ICVLHandmodel(args.batch_size, args.gpu_id)
            elif args.dataset =='msra':
                self.handmodelLayer = MSRAHandmodel(args.batch_size, args.gpu_id)
            self.head = nn.Sequential(_make_layer(BasicBlock, 256, 512, 2, stride=2),
                                      nn.AvgPool2d(4, stride=1))


        # self.fc_pos0 = nn.Linear(1024, args.joint_num * 3)

        self.fc_pos = nn.Linear(1024, args.joint_num * 3)
        self.fc_theta = nn.Linear(512, args.model_para)

    def forward(self, x):
        out, feature = self.res(x)
        pos0 = self.fc_pos(out.view(out.size(0), -1))
        c2, c3, c4, c5 = feature
        # muti super
        out = self.head(c4)
        theta = self.fc_theta(out.view(out.size(0), -1))
        pos1 = self.handmodelLayer.calculate_position(theta)
        # theta = self.fc_pos0(out)
        # pos0 = self.handmodelLayer.calculate_position(theta)
        # heatmap = self.GFM_.joint2heatmap2d(pos0.view(pos0.size(0),-1,3))
        # depthmap = self.GFM_.depth2map(pos0.view(pos0.size(0),-1,3)[:,:,2])
        # out = self.head(c1, x, pos0)
        # self.handmodelLayer.calculate_mesh()
        # mesh_img = self.handmodelLayer.mesh2pcl(128)
        # pos1 = self.head(mesh_img)
        return pos0, pos1


class DAHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAHead, self).__init__()
        inter_channels = in_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


class maskHead(nn.Module):
    def __init__(self, in_channels):
        super(maskHead, self).__init__()
        self.fusion = _make_layer(BasicBlock, in_channels, in_channels, 2)



    def forward(self, x, mask):
        batchsize, C, f_height, f_width = x.size()
        batchsize, _, img_height, img_width = mask.size()
        mask = nn.functional.max_pool2d(mask, 3, stride=(img_height/f_height))
        mask = mask.expand(batchsize, C, f_height, f_width)
        feat1 = x * mask
        out = self.fusion(feat1)
        return x + out


class heatmapHead(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(heatmapHead, self).__init__()
        self.fusion = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1)
        self.layer4 = _make_layer(BasicBlock, out_channels//2, out_channels, 2, stride=2)

    def forward(self, x, heatmap,depthmap):
        batchsize, C, f_height, f_width = x.size()
        batchsize, _, img_height, img_width = heatmap.size()
        heatmap = nn.functional.avg_pool2d(heatmap, 3, stride=(img_height/f_height))
        depth = nn.functional.avg_pool2d(depthmap, 3, stride=(img_height/f_height))
        heat_depth = heatmap*depth
        heatmap = torch.sum(heatmap, dim=1).view(batchsize, 1, f_height, f_width)
        heatmap = heatmap.expand(batchsize, C, f_height, f_width)
        out = x * heatmap
        out = torch.cat((out,heat_depth),dim=1)
        out = self.fusion(out)
        out = out + x
        out = self.layer4(out)
        return out


class edtmapHead(nn.Module):
    def __init__(self, in_channels, out_channels, GFM):
        super(edtmapHead, self).__init__()
        self.GFM = GFM
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.fusion2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, kernel_size=1)
        self.fusion3 = nn.Conv2d(in_channels=out_channels // 4 + self.GFM.J, out_channels=out_channels // 4, kernel_size=1)
        self.fusion4 = nn.Conv2d(in_channels=out_channels // 2 + self.GFM.J, out_channels=out_channels // 2, kernel_size=1)
        self.layer2 = _make_layer(BasicBlock, out_channels // 8, out_channels // 4, 2, stride=2)
        self.layer3 = _make_layer(BasicBlock, out_channels // 4, out_channels // 2, 2, stride=2)
        self.layer4 = _make_layer(BasicBlock, out_channels // 2, out_channels, 2, stride=2)


    def forward(self, x, img, joint):
        batchsize, C, f_height, f_width = x.size()
        edtmap = self.GFM.imgs2edt2(img, joint, f_height)
        distmap = self.GFM.imgs2dist(img, joint, f_height)


        edt2_2 = edtmap*distmap
        edt2_3 = self.avgpool(edt2_2)
        edt2_4 = self.avgpool(edt2_3)

        fusion2 = torch.cat((x, edt2_2), dim=1)
        fusion2 = self.fusion2(fusion2)
        out = fusion2 + x
        out = self.layer2(out)

        fusion3 = torch.cat((out, edt2_3), dim=1)
        fusion3 = self.fusion3(fusion3)
        out = fusion3 + out
        out = self.layer3(out)

        fusion4 = torch.cat((out, edt2_4), dim=1)
        fusion4 = self.fusion4(fusion4)
        out = fusion4 + out
        out = self.layer4(out)
        return out


class remapHead(nn.Module):
    def __init__(self, in_channels, out_channels, joint_num,feature_size):
        super(remapHead, self).__init__()
        self.feature_size =feature_size
        self.remap = nn.Linear(joint_num*3, in_channels//8*feature_size*feature_size)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.fusion = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels//8 * 2, out_channels=in_channels, kernel_size=1)
        self.layer4 = _make_layer(BasicBlock, in_channels, out_channels, 2, stride=2)


    def forward(self, x, joint):
        batchsize = joint.size(0)
        joint_map = self.remap(joint.view(joint.size(0),-1)).view(batchsize,-1,self.feature_size,self.feature_size)
        out = self.conv1(x)
        out = torch.cat((out,joint_map),dim=1)
        out = self.conv2(out)
        out = out + x
        out = self.layer4(out)
        return out


class meshHead(nn.Module):
    def __init__(self, img_size, joint_num):
        super(meshHead, self).__init__()
        self.img_size = img_size

        self.res = resnet18(pretrained=False)
        num_ftrs = self.res.fc.in_features
        self.res.fc = nn.Linear(num_ftrs, joint_num*3)


    def forward(self, x):
        out, features = self.res(x)
        return out


class voteHead(nn.Module):
    def __init__(self, in_channels,joint_num):
        super(voteHead, self).__init__()
        self.smooth = _make_layer(BasicBlock, in_channels, in_channels * 2, 2, stride=1)
        self.res_pos = _make_layer(BasicBlock, in_channels * 2, in_channels * 2, 2, stride=1)
        self.conv_pos = nn.Conv2d(in_channels=in_channels * 2, out_channels=3*joint_num, kernel_size=1)
        self.res_weight = _make_layer(BasicBlock, in_channels * 2, in_channels * 2, 2, stride=1)
        self.conv_weight = nn.Conv2d(in_channels=in_channels * 2, out_channels=3*joint_num, kernel_size=1)
        self.sg = nn.Sigmoid()
        self.joint_num = joint_num

    def forward(self, x):
        out = self.smooth(x)
        weight = self.res_weight(out)
        weight = self.conv_weight(weight)
        weight = self.sg(weight)
        pos = self.res_pos(out)
        pos = self.conv_pos(pos)
        pos_weight = pos * weight
        pos = pos_weight.sum(dim=-1).sum(dim=-1) / (weight.sum(-1).sum(-1)) #B*(J*3)
        return pos, weight


class joint_decoder(nn.Module):
    def __init__(self, joint_num, output_channel):
        super(joint_decoder, self).__init__()
        self.joint_num = joint_num
        self.relu = nn.ReLU(inplace=True)
        # self.fc_1 = nn.Sequential(nn.Linear(3*joint_num, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        # self.fc_2 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        # self.fc_3 = nn.Sequential(nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        # self.uppool1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        # self.uppool2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        # self.uppool3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        # self.uppool4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        # self.uppool5 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        inchannel =512
        self.uppool0 = nn.Sequential(nn.ConvTranspose2d(joint_num*3, inchannel, kernel_size=4, stride=4), nn.BatchNorm2d(inchannel),nn.ReLU())
        self.uppool1 = nn.Sequential(nn.ConvTranspose2d(inchannel, inchannel, kernel_size=2, stride=2), nn.BatchNorm2d(inchannel), nn.ReLU())
        self.uppool2 = nn.Sequential(nn.ConvTranspose2d(inchannel, inchannel/2, kernel_size=2, stride=2), nn.BatchNorm2d(inchannel/2), nn.ReLU())
        inchannel = inchannel / 2
        self.uppool3 = nn.Sequential(nn.ConvTranspose2d(inchannel, inchannel/2, kernel_size=2, stride=2), nn.BatchNorm2d(inchannel/2), nn.ReLU())
        inchannel = inchannel / 2
        self.uppool4 = nn.Sequential(nn.ConvTranspose2d(inchannel, inchannel/2, kernel_size=2, stride=2), nn.BatchNorm2d(inchannel/2), nn.ReLU())
        inchannel = inchannel / 2
        self.uppool5 = nn.Sequential(nn.ConvTranspose2d(inchannel, output_channel, kernel_size=2, stride=2))

        # for edt
        # self.uppool1 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.Dropout2d(0.1, False), nn.ReLU())
        # self.uppool2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.Dropout2d(0.1, False), nn.ReLU())
        # self.uppool3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.Dropout2d(0.1, False), nn.ReLU())


        # self.conv_depth = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)#32*128*128
        # self.bn_conv_depth = nn.BatchNorm2d(32)
        # self.layer1 = _make_layer(BasicBlock, 32, 32, 2, stride=2)#32*32*32
        # self.fusion1 = nn.Conv2d(64, 64, kernel_size=1) #64*32*32
        # self.fusion2 = nn.Conv2d(64, output_dim, kernel_size=1)  # 1*32*32

    def forward(self, x, img):
        j_out = x.view(x.size(0), self.joint_num*3, 1, 1)
        # j_out = self.fc_1(j_out)
        # j_out = self.fc_2(j_out)
        # j_out = self.fc_3(j_out)
        # j_out = j_out.view(j_out.size(0), 128, 4, 4)
        j_out = self.uppool0(j_out)
        j_out = self.uppool1(j_out)
        j_out = self.uppool2(j_out)
        j_out = self.uppool3(j_out)
        j_out = self.uppool4(j_out)
        j_out = self.uppool5(j_out)

        # i_out = self.conv_depth(img)
        # i_out = self.bn_conv_depth(i_out)
        # i_out = self.relu(i_out)
        # i_out = self.layer1(i_out)
        #
        # fusion = torch.cat((j_out, i_out), dim=1)

        # fusion = j_out
        # fusion = j_out
        # fusion = self.fusion1(fusion)
        # fusion = self.fusion2(fusion)

        return j_out


class joint_decoder_DC(nn.Module):
    def __init__(self, joint_num, output_channel):
        super(joint_decoder_DC, self).__init__()
        nf = 64
        self.joint_num = joint_num
        # Decoder
        self.convt1 = nn.ConvTranspose2d((joint_num * 3),
                                         #        self.convt1 = nn.ConvTranspose2d(num_prior_dim,
                                         (nf * 8), (4, 4),
                                         stride=1, padding=0, bias=False)
        self.bn1_d = nn.BatchNorm2d(nf * 8)
        self.convt2 = nn.ConvTranspose2d((nf * 8), (nf * 4), (4, 4),
                                         stride=2, padding=1, bias=False)
        self.bn2_d = nn.BatchNorm2d(nf * 4)
        self.convt3 = nn.ConvTranspose2d((nf * 4), (nf * 2), (4, 4),
                                         stride=2, padding=1, bias=False)
        self.bn3_d = nn.BatchNorm2d(nf * 2)
        self.convt4 = nn.ConvTranspose2d((nf * 2), nf, (4, 4),
                                         stride=2, padding=1, bias=False)
        self.bn4_d = nn.BatchNorm2d(nf)
        self.convt5 = nn.ConvTranspose2d(nf, nf, (4, 4),
                                         stride=2, padding=1, bias=False)
        self.bn5_d = nn.BatchNorm2d(nf)

        self.convt6 = nn.ConvTranspose2d(nf, 1, (4, 4),
                                         stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, img):
        z = x.view(x.size(0), self.joint_num*3, 1, 1)
        y = self.leakyrelu(self.bn1_d(self.convt1(z)))
        y = self.leakyrelu(self.bn2_d(self.convt2(y)))
        y = self.leakyrelu(self.bn3_d(self.convt3(y)))
        y = self.leakyrelu(self.bn4_d(self.convt4(y)))
        y = self.leakyrelu(self.bn4_d(self.convt5(y)))
        y = torch.tanh(self.convt6(y))
        return y

def basic_model(model_name, num_classes, use_pretrained = False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "seresnet18":
        """ SEResnet18
        """
        model_ft = seresnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "seresnet50":
        """ SEResnet50
        """
        model_ft = seresnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
    # return model_ft, input_size


def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)



if __name__ == '__main__':
    batch_size = 128
    img_size = 128
    joint_num = 14
    torch.cuda.set_device(1)
    joint = torch.rand([batch_size,joint_num*3]).cuda()
    img = torch.rand([batch_size,1,img_size,img_size]).cuda()
    net = basic_model('resnet18',joint_num*3).cuda()
    # net = linear_D(joint_num).cuda()
    output, _ = net(img)
    print(output.size())