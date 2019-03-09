from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm
from module_2d_se import SEBasicBlock
from model_3d import Pool3DBlock, Res3DBlock, Upsample3DBlock, Basic3DBlock,EncoderDecorder,AvgPool3DBlock
from module_point import Input_transform_net, Feature_transform_net, conv_bn, fc_bn
from module_pointplus import PointNetSetAbstraction, PointNetFeaturePropagation
from SO_ae import Decoder, DecoderLinear
import sys
sys.path.append('/data/users/pfren/pycharm/pointnet2')
import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class D_SA(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, input_dim, image_size=64, conv_dim=64, linear=False):
        super(D_SA, self).__init__()
        self.imsize = image_size
        self.linear = linear
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(input_dim, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        curr_size = image_size / 2

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        curr_size = curr_size / 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        curr_size = curr_size / 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
            curr_size = curr_size / 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if linear:
            last.append(nn.AvgPool2d(curr_size))
            self.fc = nn.Linear(curr_dim, 256)
        else:
            last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(curr_dim, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        if self.imsize == 64:
           out = self.l4(out)
        out, p2 = self.attn2(out)
        if self.linear:
            out = self.last(out)
            out = self.fc(out.view(out.size(0), -1))
        else:
            out = self.last(out)

        return out.squeeze(), p1, p2


class D_basic(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, input_dim, image_size=64, conv_dim=64, linear=False):
        super(D_basic, self).__init__()
        self.imsize = image_size
        self.linear = linear
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(input_dim, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        curr_size = image_size / 2

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        curr_size = curr_size / 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        curr_size = curr_size / 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
            curr_size = curr_size / 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if linear:
            last.append(nn.AvgPool2d(curr_size))
            self.fc = nn.Linear(curr_dim, 256)
        else:
            last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        if self.imsize == 64:
           out = self.l4(out)
        if self.linear:
            out = self.last(out)
            out = self.fc(out.view(out.size(0), -1))
        else:
            out = self.last(out)

        return out.squeeze(), None, None


class D_SA_M(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, dim_list, image_size_list, dis_list):
        super(D_SA_M, self).__init__()
        self.feature_num = len(dis_list)
        self.model_list = []
        for index, dis_name in enumerate(dis_list):
            if 'SA' in dis_name:
                self.model_list.append(D_SA(dim_list[index], image_size=image_size_list[index], linear=True).cuda())
            elif 'basci' in dis_name:
                self.model_list.append(D_basic(dim_list[index], image_size=image_size_list[index], linear=True).cuda())
            elif 'point_plus' in dis_name:
                self.model_list.append(D_point_plus(dim_list[index], ball_radius=0.04, ball_radius2=0.12, output_dim=256).cuda())
        self.fc = nn.Linear(self.feature_num*256, 1)

    def forward(self, x):
        p1_list = []
        p2_list = []
        output, p1, p2 = self.model_list[0](x[0])
        p1_list.append(p1)
        p2_list.append(p2)
        for index in range(1, self.feature_num):
            output_temp, p1_temp, p2_temp = self.model_list[index](x[index])
            output = torch.cat((output, output_temp), dim=-1)
            p1_list.append(p1_temp)
            p2_list.append(p2_temp)
        output = self.fc(output)
        return output.squeeze(), p1_list, p2_list


class D_HG(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, J):
        super(D_HG, self).__init__()
        self.conv = nn.Conv2d(inchannels, channels/2, kernel_size=7, stride=1, padding=3)
        self.res1 = SEBasicBlock(channels/2, channels)
        self.res2 = SEBasicBlock(channels,  channels)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res1_e = SEBasicBlock(channels, channels)
        self.res2_e = SEBasicBlock(channels, channels)
        self.res3_e = SEBasicBlock(channels, channels)
        self.res1_d = SEBasicBlock(channels, channels)
        self.res2_d = SEBasicBlock(channels, channels)
        self.res3_d = SEBasicBlock(channels, channels)
        # self.uppool1 = nn.ConvTranspose2d(channels, channels, kernel_size=2,stride=2)
        # self.uppool2 = nn.ConvTranspose2d(channels, channels, kernel_size=2,stride=2)
        # self.uppool3 = nn.ConvTranspose2d(channels, channels, kernel_size=2,stride=2)
        self.res3_c = SEBasicBlock(channels, channels)
        self.res2_c = SEBasicBlock(channels, channels)
        self.res1_c = SEBasicBlock(channels, channels)
        self.res0_c = SEBasicBlock(channels, channels)

        self.res = SEBasicBlock(channels, channels)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.cov_heat = nn.Conv2d(channels, J, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.J = J

    def forward(self, x):
        ## x:64 res1:32 res2:16 res3:8
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)

        res1 = self.pool1(out)
        res0_c = self.res0_c(out)
        res1 = self.res1_e(res1)


        res2 = self.pool2(res1)
        res1_c = self.res1_c(res1)
        res2 = self.res2_e(res2)

        res3 = self.pool3(res2)
        res2_c = self.res2_c(res2)

        res3 = self.res3_e(res3)
        res3 = self.res3_c(res3)

        res3_out = self.res3_d(res3)
        # res3_out = self.uppool3(res3_out)
        res3_out = F.interpolate(res3_out, scale_factor=2)

        res3_out += res2_c

        res2_out = self.res2_d(res3_out)
        # res2_out = self.uppool2(res2_out)
        res2_out = F.interpolate(res2_out, scale_factor=2)
        res2_out += res1_c

        res1_out = self.res1_d(res2_out)
        # res1_out = self.uppool1(res1_out)
        res1_out = F.interpolate(res1_out, scale_factor=2)
        res1_out += res0_c

        y = self.res(res1_out)
        y = self.fc(y)

        heat = self.cov_heat(y)
        return heat


# class D_linear(nn.Module):
#     def __init__(self, joint_num):
#         super(D_linear, self).__init__()
#         self.encode_lays = nn.Sequential(
#             nn.Linear(3 * joint_num, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2*joint_num)
#         )
#         self.decode_lays = nn.Sequential(
#             nn.Linear(2 * joint_num, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 3*joint_num)
#         )
#
#     def forward(self, x):
#         x = x.view(x.size(0),-1)
#         x = self.encode_lays(x)
#         x = self.decode_lays(x)
#         return x

class D_linear(nn.Module):
    def __init__(self, joint_num):
        super(D_linear, self).__init__()
        nn.Linear(3 * joint_num, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class D_HG3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(D_HG3D, self).__init__()
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 32, 7),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )
        self.encoder_decoder = EncoderDecorder()
        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class D_SA3D(nn.Module):
    def __init__(self, input_channels):
        super(D_SA3D, self).__init__()
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 32, 7),
            Pool3DBlock(2),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)
        self.avg_pool = AvgPool3DBlock(4)
        self.fc = nn.Linear(128,1)
        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        x = self.mid_res(x)
        x = self.avg_pool(x)
        x = self.fc(x.view(x.size(0),-1))
        return x, x, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class D_point(nn.Module):

    def __init__(self, sample_point, K=64):
        '''
        input: B x N x 3
        output: B x 1
        '''
        super(D_point, self).__init__()

        self.num_point = sample_point
        self.K = K

        self.input_transform_net = Input_transform_net(self.num_point)
        self.feat_transform_net = Feature_transform_net(self.num_point, self.K)
        self.conv1 = conv_bn(1, 64, [1, 3])
        self.conv2 = conv_bn(64, self.K, [1, 1])
        self.conv3 = conv_bn(self.K, 64, [1, 1])
        self.conv4 = conv_bn(64, 128, [1, 1])
        self.conv5 = conv_bn(128, 1024, [1, 1])
        self.fc1 = fc_bn(1024, 512)
        self.fc2 = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, 1)

        # self.I = nn.Parameter(torch.tensor(np.eye(config.K), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.initialize_weights()

    def forward(self, x):

        B = x.size()[0]
        input_transform = self.input_transform_net(x)
        x = torch.matmul(x, input_transform)
        x = x.view((B, 1) + x.size()[1:])
        x = self.conv1(x)
        x = self.conv2(x)

        feat_transform = self.feat_transform_net(x)

        x = x.squeeze(3).permute(0, 2, 1).contiguous()
        x = torch.matmul(x, feat_transform)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size() + (1,))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Npymmetric function: max pooling
        x = F.max_pool2d(x, [self.num_point, 1])
        x = x.view([B, -1])
        x = self.fc1(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.dropout(x, p=0.3)
        x = self.fc3(x)

        return x,x,x

    def initialize_weights(self):
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
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class D_point_plus(nn.Module):
    def __init__(self, input_dim, ball_radius=0.015, ball_radius2=0.04, knn_K=64, output_dim=1):
        super(D_point_plus, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, ball_radius, knn_K, input_dim, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, ball_radius2, knn_K, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz[:, 0:3, :], xyz[:, 3:, :])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, x, x


class D_point_plus_seg(nn.Module):
    def __init__(self,out_dim, input_dim=3,  ball_radius=0.015, ball_radius2=0.04, knn_K=64):
        super(D_point_plus_seg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, ball_radius, knn_K, input_dim, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, ball_radius2, knn_K, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 1024], True)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_dim, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz[:, 0:3, :], xyz[:, 3:, :])#l1_xyz is position l1_points is the feature
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz[:, 0:3, :], l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return x

class D_point_plus_seg_complex(nn.Module):
    def __init__(self, num_classes,ball_radius_list=[0.015,0.04,0.08,0.16]):
        super(D_point_plus_seg_complex, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, ball_radius_list[0], 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256,ball_radius_list[1], 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, ball_radius_list[2], 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, ball_radius_list[3], 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        return x

#auto_encoder only fc
class D_point_AE_fc(nn.Module):
    def __init__(self, input_dim, out_dim, dim_list=[512,128], ball_radius=0.015, ball_radius2=0.04, knn_K=64):
        super(D_point_AE_fc, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, ball_radius, knn_K, input_dim, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, ball_radius2, knn_K, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 1024], True)
        # self.decoder = DecoderLinear(1024, out_dim)
        self.upfc1 = nn.Sequential(nn.Linear(1024, dim_list[0]), nn.BatchNorm1d(dim_list[0]), nn.ReLU())
        self.upfc2 = nn.Sequential(nn.Linear(dim_list[0], dim_list[1]), nn.BatchNorm1d(dim_list[1]), nn.ReLU())
        self.upfc3 = nn.Sequential(nn.Linear(dim_list[1], out_dim))


    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz[:, 0:3, :], xyz[:, 3:, :])#l1_xyz is position l1_points is the feature
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, 1024)
        # x = self.decoder(x)
        x = self.upfc1(x)
        x = self.upfc2(x)
        x = self.upfc3(x)
        x = x.view(x.size(0), 3, -1)
        return x, l2_points, l3_points

#auto_encoder
class D_point_AE_fc_conv(nn.Module):
    def __init__(self, input_dim, gpu_id, ball_radius=0.015, ball_radius2=0.04, knn_K=64):
        super(D_point_AE_fc_conv, self).__init__()
        torch.cuda.set_device(gpu_id)
        self.sa1 = PointNetSetAbstraction(512, ball_radius, knn_K, input_dim, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, ball_radius2, knn_K, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 1024], True)
        self.decoder = Decoder(1024, 256, 256).cuda()

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz[:, 0:3, :], xyz[:, 3:, :])#l1_xyz is position l1_points is the feature
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, 1024)
        out = self.decoder(x)
        return out, l2_xyz, l3_xyz


class Pointnet2SSG_seg(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True, ball_radius_list=[0.04, 0.08, 0.16, 0.32], knn_K=64):
        super(Pointnet2SSG_seg, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=ball_radius_list[0],
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=ball_radius_list[1],
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=ball_radius_list[2],
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=ball_radius_list[3],
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz))

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))


        self.FC_layer = nn.Sequential(nn.Conv1d(128, 128, 1),nn.BatchNorm1d(128),nn.Dropout(0.5),nn.Conv1d(128, num_classes, 1))
        # self.FC_layer = (pt_utils.Seq(128).conv1d(128, bn=True).dropout().conv1d(num_classes, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous()
                    if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        pointcloud = pointcloud.permute(0, 2, 1)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        return self.FC_layer(l_features[0])


class Pointnet2SSG_cls(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True, ball_radius=0.015, ball_radius2=0.04):
        super(Pointnet2SSG_cls, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=ball_radius,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=ball_radius2,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz))
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz))

        if num_classes > 100:
            self.FC_layer = (pt_utils.Seq(1024) \
                             .fc(2048, bn=True)
                             .dropout(0.5)
                             .fc(2048, bn=True)
                             .dropout(0.5)
                             .fc(num_classes, activation=None))
        else:
            self.FC_layer = (pt_utils.Seq(1024) \
                             .fc(512, bn=True)
                             .dropout(0.5)
                             .fc(256, bn=True)
                             .dropout(0.5)
                             .fc(num_classes, activation=None))



    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous()
                    if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        pointcloud = pointcloud.permute(0,2,1)
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))

class Pointnet2SSG_simple(nn.Module):

    def __init__(self, num_classes, input_channels=3, use_xyz=True, ball_radius_list=[0.04, 0.08, 0.16, 0.32], knn_K=64):
        super(Pointnet2SSG_simple, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=ball_radius_list[0],
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=ball_radius_list[1],
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1,
                radius=2.0,
                nsample=64,
                mlp=[256, 256, 512, 1024],
                use_xyz=use_xyz)
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[1024 + 256, 256, 256]))


        self.FC_layer = nn.Sequential(nn.Conv1d(128, 128, 1),nn.BatchNorm1d(128),nn.Dropout(0.5),nn.Conv1d(128, num_classes, 1))
        # self.FC_layer = (pt_utils.Seq(128).conv1d(128, bn=True).dropout().conv1d(num_classes, activation=None))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous()
                    if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        return self.FC_layer(l_features[0])


# ge's network
from point_util import group_points_2
from point_util import group_points
from point_util import propagation_points
nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,512]

nstates_FP_3 = [1280,512,256]
nstates_FP_2 = [384,128,128]
nstates_FP_1 = [128,128,128]

class PointNet_Plus_seg(nn.Module):
    def __init__(self,  output_dim, input_dim, sample_num, ball_radius,ball_radius2, knn_K = 64):
        super(PointNet_Plus_seg, self).__init__()
        self.knn_K = knn_K
        self.ball_radius = ball_radius
        self.ball_radius2 = ball_radius2
        self.sample_num_level1 = 512
        self.sample_num_level2 = 128
        self.INPUT_FEATURE_NUM = input_dim
        self.SAMPLE_NUM = sample_num
        self.num_outputs = output_dim

        self.netR_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )

        self.netR_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )

        self.netR_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )
        self.netFP_3 = nn.Sequential(
            # B*(3 + 256 + 1024 )*sample_num_level2*1
            nn.Conv2d(3 + nstates_FP_3[0], nstates_FP_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[1]),
            nn.ReLU(inplace=True),
            # B*(256)*sample_num_level2*1
            nn.Conv2d(nstates_FP_3[1], nstates_FP_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_2 = nn.Sequential(
            # B*(3 + 128 + 256)*sample_num_level1*1
            nn.Conv2d(3 + nstates_FP_2[0],nstates_FP_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[1]),
            nn.ReLU(inplace=True),
            # B*(128)*sample_num_level2*1
            nn.Conv2d(nstates_FP_2[1], nstates_FP_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_1 = nn.Sequential(
            # B*(3+3 +128)*SAMPLE_NUM*1
            nn.Conv2d(3 + nstates_FP_1[0],nstates_FP_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[1]),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(nstates_FP_1[1], nstates_FP_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[2]),
            nn.ReLU(inplace=True),
        )
        self.net_seg = nn.Sequential(
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(128,128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(128, self.num_outputs, kernel_size=(1, 1))
        )

    def forward(self, points):
        points = points.permute(0, 2, 1)
        points_xyz = points[:,:,0:3]
        # points: B * 1024 * 6
        inputs_level1, inputs_level1_center = group_points(points, self.sample_num_level1, self.SAMPLE_NUM, self.knn_K, self.ball_radius, self.INPUT_FEATURE_NUM)
        outputs_level1 = self.netR_1(inputs_level1)
        # B*128*sample_num_level1*1
        outputs_level1 = torch.cat((inputs_level1_center, outputs_level1),1).squeeze(-1)
        # B*(3+128)*sample_num_level1

        inputs_level2, inputs_level2_center = group_points_2(outputs_level1, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        # B*131*sample_num_level2*knn_K
        outputs_level2 = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        outputs_level2 = torch.cat((inputs_level2_center, outputs_level2),1)
        # B*259*sample_num_level2*1

        outputs_level3 = self.netR_3(outputs_level2)
        # B*1024*1*1 feature propagation
        outputs_level3 = outputs_level3.expand(-1, -1,self.sample_num_level2,-1)
        # B*1024*sample_num_level2*1
        res_level3 =torch.cat((outputs_level2, outputs_level3),1)
        # B*(3+256+1024)*sample_num_level2*1
        res_level3 = self.netFP_3(res_level3)
        # B*(256)*sample_num_level2*1
        res_level2 = propagation_points(res_level3.squeeze(-1), outputs_level1, self.sample_num_level2, self.sample_num_level1,3)
        # B*(3+128+256)*sample_num_level1*1
        res_level2 = self.netFP_2(res_level2)
        # B*(128)*sample_num_level1*1
        res_level1 = propagation_points(res_level2.squeeze(-1), points_xyz.transpose(1,2), self.sample_num_level1,self.SAMPLE_NUM,3)
        # B*(3+3+128)*SAMPLE_NUM*1
        res_level1 = self.netFP_1(res_level1)
        # B*(128)*sample_num_level1*1
        res_level1 = self.net_seg(res_level1)
        # B*(4*J)*sample_num_level1*1
        return res_level1.squeeze(-1)

class PointNet_Plus_seg_two(nn.Module):
    def __init__(self, joint_num, sample_num, ball_radius,ball_radius2, knn_K = 64):
        super(PointNet_Plus_seg_two, self).__init__()
        self.num_outputs = joint_num*3
        self.knn_K = knn_K
        self.ball_radius = ball_radius
        self.ball_radius2 = ball_radius2
        self.sample_num_level1 = 512
        self.sample_num_level2 = 128
        self.INPUT_FEATURE_NUM = 3
        self.SAMPLE_NUM = sample_num
        self.JOINT_NUM = joint_num

        self.netR_1_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )

        self.netR_1_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )

        self.netR_1_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )
        self.netFP_1_3 = nn.Sequential(
            # B*(3 + 256 + 1024 )*sample_num_level2*1
            nn.Conv2d(3 + nstates_FP_3[0], nstates_FP_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[1]),
            nn.ReLU(inplace=True),
            # B*(256)*sample_num_level2*1
            nn.Conv2d(nstates_FP_3[1], nstates_FP_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_1_2 = nn.Sequential(
            # B*(3 + 128 + 256)*sample_num_level1*1
            nn.Conv2d(3 + nstates_FP_2[0],nstates_FP_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[1]),
            nn.ReLU(inplace=True),
            # B*(128)*sample_num_level2*1
            nn.Conv2d(nstates_FP_2[1], nstates_FP_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_1_1 = nn.Sequential(
            # B*(3 + 128)*SAMPLE_NUM*1
            nn.Conv2d(3 + nstates_FP_1[0],nstates_FP_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[1]),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(nstates_FP_1[1], nstates_FP_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[2]),
            nn.ReLU(inplace=True),
        )
        self.net_1_seg = nn.Sequential(
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(128,128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(128, self.JOINT_NUM*4, kernel_size=(1, 1))
        )
        self.netR_2_1 = nn.Sequential(
            # B*(INPUT_FEATURE_NUM+nstates_FP_1[2]+JOINT_NUM*4)*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM + nstates_FP_1[2] + self.JOINT_NUM*4, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, self.knn_K), stride=1)
            # B*128*sample_num_level1*1
        )

        self.netR_2_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3 + nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1, self.knn_K), stride=1)
            # B*256*sample_num_level2*1
        )

        self.netR_2_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2, 1), stride=1),
            # B*1024*1*1
        )
        self.netFP_2_3 = nn.Sequential(
            # B*(3 + 256 + 1024 )*sample_num_level2*1
            nn.Conv2d(3 + nstates_FP_3[0], nstates_FP_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[1]),
            nn.ReLU(inplace=True),
            # B*(256)*sample_num_level2*1
            nn.Conv2d(nstates_FP_3[1], nstates_FP_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_3[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_2_2 = nn.Sequential(
            # B*(3 + 128 + 256)*sample_num_level1*1
            nn.Conv2d(3 + nstates_FP_2[0], nstates_FP_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[1]),
            nn.ReLU(inplace=True),
            # B*(128)*sample_num_level2*1
            nn.Conv2d(nstates_FP_2[1], nstates_FP_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_2[2]),
            nn.ReLU(inplace=True),
        )
        self.netFP_2_1 = nn.Sequential(
            # B*(INPUT_FEATURE_NUM + 128 + J*4 +128)*SAMPLE_NUM*1
            nn.Conv2d(self.INPUT_FEATURE_NUM+2*nstates_FP_2[2] + self.JOINT_NUM*4, nstates_FP_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[1]),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(nstates_FP_1[1], nstates_FP_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[2]),
            nn.ReLU(inplace=True),
        )
        self.net_2_seg = nn.Sequential(
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(nstates_FP_1[2], nstates_FP_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_FP_1[2]),
            nn.ReLU(inplace=True),
            # B*(128)*SAMPLE_NUM*1
            nn.Conv2d(128, self.JOINT_NUM * 4, kernel_size=(1, 1))
        )

        self.netR_FC = nn.Sequential(
            # B*1024
            nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )
    def forward(self, points):
        # points: B * 1024 * INPUT_FEATURE_NUM
        inputs_level1, inputs_level1_center = group_points(points, self.sample_num_level1, self.SAMPLE_NUM, self.knn_K, self.ball_radius, self.INPUT_FEATURE_NUM)
        outputs_level1 = self.netR_1_1(inputs_level1)
        # B*128*sample_num_level1*1
        outputs_level1 = torch.cat((inputs_level1_center, outputs_level1),1).squeeze(-1)
        # B*(3+128)*sample_num_level1

        inputs_level2, inputs_level2_center = group_points_2(outputs_level1, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        # B*131*sample_num_level2*knn_K
        outputs_level2 = self.netR_1_2(inputs_level2)
        # B*256*sample_num_level2*1
        outputs_level2 = torch.cat((inputs_level2_center, outputs_level2),1)
        # B*259*sample_num_level2*1

        outputs_level3 = self.netR_1_3(outputs_level2)
        # B*1024*1*1 feature propagation
        outputs_level3 = outputs_level3.expand(-1, -1,self.sample_num_level2,-1)
        # B*1024*sample_num_level2*1
        res_level3 =torch.cat((outputs_level2, outputs_level3),1)
        # B*(3+256+1024)*sample_num_level2*1
        res_level3 = self.netFP_1_3(res_level3)
        # B*(256)*sample_num_level2*1
        res_level2 = propagation_points(res_level3.squeeze(-1), outputs_level1, self.sample_num_level2, self.sample_num_level1,3)
        # B*(3+128+256)*sample_num_level1*1
        res_level2 = self.netFP_1_2(res_level2)
        # B*(128)*sample_num_level1*1
        res_level1 = propagation_points(res_level2.squeeze(-1), points.transpose(1,2), self.sample_num_level1,self.SAMPLE_NUM,3)
        # B*(3+3+128)*SAMPLE_NUM*1
        res_level1 = self.netFP_1_1(res_level1)
        # B*(128)*sample_num_level1*1
        res_level1_s1 = self.net_1_seg(res_level1).squeeze(-1)
        # B*(4*J)*sample_num_level1


        #stage 2
        stage_2_input_points = torch.cat((points, res_level1.squeeze(-1).transpose(1,2),res_level1_s1.transpose(1,2)),2)
        # points: B * 1024 * (6+128+4*j)
        inputs_level1, inputs_level1_center = group_points(stage_2_input_points, self.sample_num_level1, self.SAMPLE_NUM, self.knn_K, self.ball_radius, (self.INPUT_FEATURE_NUM+nstates_FP_1[2]+self.JOINT_NUM*4))
        outputs_level1 = self.netR_2_1(inputs_level1)
        # B*128*sample_num_level1*1
        outputs_level1 = torch.cat((inputs_level1_center, outputs_level1),1).squeeze(-1)
        # B*(3+128)*sample_num_level1

        inputs_level2, inputs_level2_center = group_points_2(outputs_level1, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        # B*131*sample_num_level2*knn_K
        outputs_level2 = self.netR_2_2(inputs_level2)
        # B*256*sample_num_level2*1
        outputs_level2 = torch.cat((inputs_level2_center, outputs_level2),1)
        # B*259*sample_num_level2*1

        outputs_level3 = self.netR_2_3(outputs_level2)
        # B*1024*1*1 feature propagation

        #regress joint pca value
        global_feature = outputs_level3.view(-1, nstates_plus_3[2])
        # B*1024
        reg = self.netR_FC(global_feature)

        outputs_level3 = outputs_level3.expand(-1, -1,self.sample_num_level2,-1)
        # B*1024*sample_num_level2*1
        res_level3 = torch.cat((outputs_level2, outputs_level3),1)
        # B*(3+256+1024)*sample_num_level2*1
        res_level3 = self.netFP_2_3(res_level3)
        # B*(256)*sample_num_level2*1
        res_level2 = propagation_points(res_level3.squeeze(-1), outputs_level1, self.sample_num_level2, self.sample_num_level1,3)
        # B*(3+128+256)*sample_num_level1*1
        res_level2 = self.netFP_2_2(res_level2)
        # B*(128)*sample_num_level1*1
        res_level1 = propagation_points(res_level2.squeeze(-1), stage_2_input_points.transpose(1,2), self.sample_num_level1,self.SAMPLE_NUM,3)
        # B*(3+3+128)*SAMPLE_NUM*1
        res_level1 = self.netFP_2_1(res_level1)
        # B*(128)*sample_num_level1*1
        res_level1_s2 = self.net_2_seg(res_level1).squeeze(-1)
        # B*(4*J)*sample_num_level1*1

        return res_level1_s1, res_level1_s2,reg

if __name__ == '__main__':
    import time
    joint_num = 14
    batch_size = 16
    image_size = 128
    pool_factor = 2
    heat_size = image_size/2
    torch.cuda.set_device(0)
    # input_size = [image_size, heat_size, joint_num]
    # input_dim = [1, joint_num, 6]

    # model = D_basic(joint_num + 1).cuda()
    # feature_dim = [joint_num + 1, 4]
    # model = D_SA_M(feature_dim, [heat_size], ['SA', 'point_plus']).cuda()
    # image = torch.rand([batch_size, 1, heat_size, heat_size])
    # heatmap = torch.rand([batch_size, joint_num, heat_size, heat_size])
    # geo = torch.rand([batch_size, 6, joint_num, joint_num])
    # point = torch.rand([batch_size, 4, 2048])
    # input_list = [torch.cat((image, heatmap), dim=1).cuda(), point.cuda()]
    # vol = torch.rand([batch_size, joint_num+1, heat_size, heat_size, heat_size])
    point = torch.rand([batch_size, 3 + joint_num * 4, 1024]).cuda()
    # model = D_point_AE_fc(3, 1).cuda()
    model = PointNet_Plus_seg(joint_num*4, 3 + joint_num * 4, 1024, 0.04, 0.08).cuda()
    timer = time.time()
    out = model(point)
    print(time.time() - timer)
    print(out.size(),out.sum())
