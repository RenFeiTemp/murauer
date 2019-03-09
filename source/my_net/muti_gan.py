import argparse
from torchvision import datasets, models, transforms
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from module_2d import *
from resnet_mur import resnet50
import sys
sys.path.append('..')
from util.argparse_helper import parse_arguments_generic
from data.NyuHandPoseDataset import NyuHandPoseMultiViewDataset,NyuHandPoseDataset #
from data.basetypes import LoaderMode, DatasetType
from util.transformations import transformPoint2D
import eval.handpose_evaluation as hape_eval
from numpy.linalg import inv
import torch.backends.cudnn as cudnn
import cv2
import time
import logging
from vis_tool import draw_pose,draw_depth_heatmap
from generateFeature import GFM



restrictedjoint_numointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
calculate2deeppiror = [0, 1, 2, 3, 4, 5, 6, 7, 13, 12, 11, 9, 10, 8]
joint_select = np.array([0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 23, 24, 25, 27, 28, 32, 30, 31])
calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]

para_weight = 0.01


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0, para_weight)
        # torch.nn.init.constant(m.bias.data,0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 0, para_weight)
        torch.nn.init.normal_(m.bias.data, 0, para_weight)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, 0, para_weight)
        torch.nn.init.normal_(m.bias.data, 0, para_weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0, para_weight)
        # torch.nn.init.normal_(m.bias.data,0,para_weight)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.constant_(m.bias.data, 0.0)
    # elif isinstance(m, nn.Linear):
    #     torch.nn.init.normal_(m.weight.data, 0, 0.001)
    #     torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_resnet(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        # m.bias.data.zero_()


def load_model(net, model_dir):
    pretrained_dict = torch.load(os.path.join(model_dir))
    model_dict = net.state_dict()
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            continue
        model_dict[name].copy_(param)


class Trainer(object):
    def __init__(self, config, args_data):

        self.root_dir = config.root_dir
        self.train_debug_dir = config.train_debug_dir
        self.test_debug_dir = config.test_debug_dir
        self.batch_size = config.batch_size
        self.load_epoch = config.load_epoch
        self.epoch_max = config.epoch_max
        self.phase = config.phase
        self.gpu_id = config.gpu_id
        self.used_device = torch.device("cuda:{}".format(self.gpu_id))

        self.finetune_dir = config.finetune_dir
        self.dataset = config.dataset
        self.test_id = config.test_id
        self.model_save = config.model_save
        self.G_type = config.G_type
        self.D_type = config.D_type
        self.adv_type = config.adv_type
        self.lambda_gp = config.lambda_gp
        self.coeff = config.coeff
        self.gamma = config.gamma
        self.lambda_k = config.lambda_k
        self.k = config.k
        self.ball_radius = config.ball_radius
        self.ball_radius2 = config.ball_radius2
        self.sample_point = config.sample_point

        self.view = config.view
        self.input_size = config.input_size
        self.joint_num = config.joint_num

        self.pool_factor = config.pool_factor
        self.feature_type = config.feature_type

        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        self.G_step_size = config.G_step_size
        self.D_step_size = config.D_step_size
        self.G_opt_type = config.G_opt_type
        self.D_opt_type = config.D_opt_type
        self.scheduler_type = config.scheduler_type
        # warm-up
        self.lr_lambda = lambda epoch: (0.33 ** max(0, 2 - epoch // 2)) if epoch < 4 else np.exp(-0.04 * epoch)

        self.feature_name_list = self.feature_type.split(',')
        self.feature_size = []
        self.feature_dim = []
        self.dis_type = []
        for feature_name in self.feature_name_list:
            if feature_name == 'GD':
                self.feature_size.append(self.joint_num)
                self.feature_dim.append(6)
                self.dis_type.append('basic')
            elif feature_name == 'heatmap' or feature_name == 'edt':
                self.feature_size.append(self.input_size/self.pool_factor)
                self.feature_dim.append(self.joint_num + 1)
                self.dis_type.append('SA')
            elif feature_name == 'heatmap3D':
                self.feature_dim.append(self.joint_num + 1)
                self.dis_type.append('SA3D')
            elif feature_name == 'point':
                self.feature_dim.append(4)
                self.dis_type.append('point_plus')

        self.data_rt = self.root_dir + "/" + self.dataset
        if self.model_save == '':
            if self.D_type !='NULL':
                self.model_save = 'gan/'+self.D_type + '_' + self.feature_type + '_' + self.adv_type +\
                              '_' + 'G'+str(self.D_opt_type) + str(self.G_lr) + '_' + 'D' +str(self.D_opt_type)+ str(self.D_lr) + "_" + self.scheduler_type\
                              + 'batch'+str(self.batch_size)
            else:
                self.model_save = self.G_type + '_' + 'G' + str(self.G_opt_type) + str(self.G_lr) + "_" + self.scheduler_type + 'batch' + str(self.batch_size)
            if self.finetune_dir != '':
                self.model_save = self.model_save + '_finetune'

        self.model_dir = './model/' + self.dataset + '/' + self.model_save
        cache_rt = './cache/'
        self.cur_epochs = self.load_epoch

        if self.finetune_dir != '':
            self.finetune_dir = './model/' + self.dataset + '/' + self.finetune_dir


        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '/img'):
            os.mkdir(self.model_dir + '/img')
        if not os.path.exists(self.train_debug_dir):
            os.mkdir(self.train_debug_dir)
        if not os.path.exists(self.test_debug_dir):
            os.mkdir(self.test_debug_dir)

        # use GPU
        self.cuda = torch.cuda.is_available()
        gpu_num = 1
        torch.cuda.set_device(self.gpu_id)
        cudnn.benchmark = True

        # set network
        if 'muti_net' in self.G_type:
            net = muti_net(config)
        elif self.G_type == 'HMM':
            net = HMM(config)
        elif self.G_type == 'JRM':
            net = JRM(config)
        elif self.G_type == 'REN':
            net = REN(self.input_size, self.joint_num * 3)
        elif self.G_type == 'MREN':
            net = MREN(config)
        elif self.G_type == 'HGR':
            net = HG_regress(128, self.joint_num)
        elif self.G_type == 'HGHM':
            net = HGHM(config, 128)
        elif self.G_type == 'HGR_stand':
            net = HG_regress_stand(32, self.joint_num)
        elif self.G_type == 'HGR_office':
            net = hg(num_stacks=2, num_blocks=4, num_classes=self.joint_num)
        elif self.G_type == 'HGR_res':
            net = HG_res(32, self.joint_num)
        elif 'FPN' in self.G_type:
            net = FPNnet(config)
        elif 'M_' in self.G_type:
            net = Mresnet(config)
        elif 'basic' in self.G_type:
            net = basic_model('resnet18', 3 * self.joint_num)
        elif 'resnet_mur' in self.G_type:
            net = resnet50(num_classes = self.joint_num*3)
        elif 'head' in self.G_type:
            net = resnet_head(config)
        elif 'G' in self.G_type:
            net = joint_decoder_DC(self.joint_num, 1)
        elif 'point_ae' == self.G_type:
            net = Pointnet2SSG_cls(self.sample_point * 3, input_channels=1, ball_radius=self.ball_radius,ball_radius2=self.ball_radius2)
            # net = Pointnet2SSG_ae(self.joint_num*3, input_channels=1)
            # net = Pointnet2SSG_cls(self.joint_num*3, input_channels=1, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
            # net = D_point_AE_fc(4, self.joint_num * 3, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
            # net = D_point_AE_fc(3, self.sample_point, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
            # net = D_point_AE_fc_conv(3,self.gpu_id,ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
            self.chamfer_criteria = ChamferLoss()
        elif 'point_seg' == self.G_type:
            # net = D_point_plus_seg(self.joint_num * 4, input_dim=3, ball_radius=self.ball_radius,ball_radius2=self.ball_radius2)
            # net = PointNet_Plus_seg(self.joint_num,self.sample_point,self.ball_radius,self.ball_radius2)
            # net = Pointnet2SSG_seg(self.joint_num*4, input_channels=self.joint_num*4)
            net = Pointnet2SSG_simple(self.joint_num * 4, input_channels=0)
        elif 'point_two_stage' == self.G_type:
            net = PointNet_Plus_seg_two(self.joint_num,self.sample_point,self.ball_radius,self.ball_radius2)
        self.G = net
        self.G.apply(weights_init_resnet)
        if gpu_num > 1:
            self.G = nn.DataParallel(self.G).cuda()
        else:
            self.G = self.G.cuda()

        if 'linear' == self.D_type:
            self.D = D_linear(self.joint_num)
            self.D.apply(weights_init_resnet)
        elif 'SA' == self.D_type:
            self.D = D_SA(self.joint_num + 1, image_size=128 / self.pool_factor)
        elif 'SA3D' == self.D_type:
            self.D = D_SA3D(self.joint_num + 1)
        elif 'SA_M' == self.D_type:
            self.D = D_SA_M(self.feature_dim, self.feature_size, self.dis_type)
            # self.D.apply(weights_init_normal)
        elif 'HG' == self.D_type:
            self.D = D_HG(self.joint_num + 1, 128, self.joint_num)
            self.D.apply(weights_init_normal)
        elif 'point' == self.D_type:
            self.D = D_point(2048)
        elif 'point_plus' == self.D_type:
            self.D = D_point_plus(4, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
        elif 'point_seg' == self.D_type:
            # self.D = Pointnet2SSG_seg(self.joint_num*4, input_channels=self.joint_num*4)
            # self.D = D_point_plus_seg(self.joint_num * 4, input_dim=3 + self.joint_num * 4 , ball_radius=self.ball_radius,ball_radius2=self.ball_radius2)
            self.D = PointNet_Plus_seg(self.joint_num*4,3 + self.joint_num * 4 ,self.sample_point, self.ball_radius, self.ball_radius2)
            self.D.apply(weights_init_resnet)
        elif 'point_reg' == self.D_type:
            self.D = Pointnet2SSG_cls(self.joint_num * 3, input_channels=1, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
            self.D.apply(weights_init_resnet)
            # self.D = D_point_AE_fc_conv(3, self.gpu_id, ball_radius=self.ball_radius, ball_radius2=self.ball_radius2)
        # for LSGAN
        self.y = torch.FloatTensor(self.batch_size).cuda()
        if self.D_type !='NULL':
            if gpu_num > 1:
                self.G = nn.DataParallel(self.G).cuda()
                self.D = nn.DataParallel(self.D).cuda()
            else:
                self.G = self.G.cuda()
                self.D = self.D.cuda()
        else:
            self.D = torch.nn.Linear(1, 1)

        # load data
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')

        # init net
        print ('init net...')
        print('  + Number of params: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

        # init loader
        if self.phase == 'train':
            if self.dataset == 'nyu':
                self.trainData = NyuHandPoseDataset(args_data.nyu_data_basepath, train=True,
                                   cropSize=args_data.in_crop_size,
                                   doJitterCom=args_data.do_jitter_com,
                                   sigmaCom=args_data.sigma_com,
                                   doAddWhiteNoise=args_data.do_add_white_noise,
                                   sigmaNoise=args_data.sigma_noise,
                                   unlabeledSampleIds=None,
                                   transform=transforms.ToTensor(),
                                   useCache=args_data.use_pickled_cache,
                                   cacheDir=args_data.nyu_data_basepath_pickled,
                                   annoType=args_data.anno_type,
                                   camIdsReal=[1],
                                   camIdsSynth=[],
                                   cropSize3D=args_data.crop_size_3d_tuple,
                                   args_data=args_data)
                self.testData = NyuHandPoseDataset(args_data.nyu_data_basepath, train=False,
                                   cropSize=args_data.in_crop_size,
                                   doJitterCom=args_data.do_jitter_com,
                                   sigmaCom=args_data.sigma_com,
                                   doAddWhiteNoise=args_data.do_add_white_noise,
                                   sigmaNoise=args_data.sigma_noise,
                                   unlabeledSampleIds=None,
                                   transform=transforms.ToTensor(),
                                   useCache=args_data.use_pickled_cache,
                                   cacheDir=args_data.nyu_data_basepath_pickled,
                                   annoType=args_data.anno_type,
                                   camIdsReal=[1],
                                   camIdsSynth=[],
                                   cropSize3D=args_data.crop_size_3d_tuple,
                                   args_data=args_data)
                # self.testData = NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=False,
                #                         cropSize=args_data.in_crop_size,
                #                         doJitterCom=args_data.do_jitter_com_test,
                #                         sigmaCom=args_data.sigma_com,
                #                         doAddWhiteNoise=args_data.do_add_white_noise_test,
                #                         sigmaNoise=args_data.sigma_noise,
                #                         transform=transforms.ToTensor(),
                #                         useCache=args_data.use_pickled_cache,
                #                         cacheDir=args_data.nyu_data_basepath_pickled,
                #                         annoType=args_data.anno_type,
                #                         neededCamIdsReal=args_data.needed_cam_ids_test,
                #                         neededCamIdsSynth=[],
                #                         randomSeed=args_data.seed,
                #                         cropSize3D=args_data.crop_size_3d_tuple,
                #                         args_data=args_data)
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.testLoader = DataLoader(self.testData, batch_size=self.batch_size, shuffle=False, num_workers=8)

        else:
            if self.dataset == 'nyu':
                # self.testData = NyuHandPoseDataset(args_data.nyu_data_basepath, train=False,
                #                    cropSize=args_data.in_crop_size,
                #                    doJitterCom=args_data.do_jitter_com,
                #                    sigmaCom=args_data.sigma_com,
                #                    doAddWhiteNoise=args_data.do_add_white_noise,
                #                    sigmaNoise=args_data.sigma_noise,
                #                    unlabeledSampleIds=None,
                #                    transform=transforms.ToTensor(),
                #                    useCache=args_data.use_pickled_cache,
                #                    cacheDir=args_data.nyu_data_basepath_pickled,
                #                    annoType=args_data.anno_type,
                #                    camIdsReal=[1],
                #                    camIdsSynth=[],
                #                    cropSize3D=args_data.crop_size_3d_tuple,
                #                    args_data=args_data)
                self.testData = NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=False,
                                        cropSize=args_data.in_crop_size,
                                        doJitterCom=args_data.do_jitter_com_test,
                                        sigmaCom=args_data.sigma_com,
                                        doAddWhiteNoise=args_data.do_add_white_noise_test,
                                        sigmaNoise=args_data.sigma_noise,
                                        transform=transforms.ToTensor(),
                                        useCache=args_data.use_pickled_cache,
                                        cacheDir=args_data.nyu_data_basepath_pickled,
                                        annoType=args_data.anno_type,
                                        neededCamIdsReal=args_data.needed_cam_ids_test,
                                        neededCamIdsSynth=[],
                                        randomSeed=args_data.seed,
                                        cropSize3D=args_data.crop_size_3d_tuple,
                                        args_data=args_data)
            self.testLoader = DataLoader(self.testData, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if self.G_opt_type == 'sgd':
            self.G_opt = optim.SGD(self.G.parameters(), lr=self.G_lr, momentum=0.9, weight_decay=1e-4)
        elif self.G_opt_type == 'adam':
            self.G_opt = optim.Adam(self.G.parameters(), lr=self.G_lr)
        elif self.G_opt_type == 'rmsprop':
            self.G_opt = optim.RMSprop(self.G.parameters(), lr=self.G_lr)

        if self.D_opt_type == 'sgd':
            self.D_opt = optim.SGD(self.D.parameters(), lr=self.D_lr, momentum=0.9, weight_decay=1e-4)
        elif self.D_opt_type == 'adam':
            self.D_opt = optim.Adam(self.D.parameters(), lr=self.D_lr)
        elif self.D_opt_type == 'rmsprop':
            self.D_opt = optim.RMSprop(self.D.parameters(), lr=self.D_lr)

        if self.scheduler_type == 'step':
            self.G_scheduler = lr_scheduler.StepLR(self.G_opt, step_size=self.G_step_size, gamma=0.1)
            self.D_scheduler = lr_scheduler.StepLR(self.D_opt, step_size=self.D_step_size, gamma=0.1)
        elif self.scheduler_type =='SGDR':
            self.G_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.G_opt, int(float(len(self.trainData))/self.batch_size))
            self.D_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.D_opt, int(float(len(self.trainData))/self.batch_size))
        elif self.scheduler_type == 'warm-up':
            self.G_scheduler = optim.lr_scheduler.LambdaLR(self.G_opt, lr_lambda=self.lr_lambda)
            self.D_scheduler = optim.lr_scheduler.LambdaLR(self.D_opt, lr_lambda=self.lr_lambda)


        # load model
        if self.load_epoch != 0:
            self.G.load_state_dict(torch.load(os.path.join(self.model_dir + '/latest_G' + str(self.load_epoch) + '.pth'),
                                              map_location=lambda storage, loc: storage))
            if self.D_type != 'NULL':
                self.D.load_state_dict(torch.load(os.path.join(self.model_dir + '/latest_D' + str(self.load_epoch) + '.pth'),
                                              map_location=lambda storage, loc: storage))

        if self.finetune_dir != '':
            self.G.load_state_dict(torch.load(os.path.join(self.finetune_dir),map_location=lambda storage, loc: storage))
        # self.D.load_state_dict(torch.load(os.path.join('./model/nyu/gan/point_plus_RaLS_finetune_lrG0.1_lrD1e-3/latest_D46.pth'),map_location=lambda storage, loc: storage))
        self.GFM_ = GFM(self.batch_size, self.gpu_id, self.input_size / self.pool_factor, self.joint_num,sample_num = self.sample_point)
        self.pcl_label = torch.zeros([self.batch_size,1,self.sample_point]).cuda()
        self.loss_list_name = []
        self.loss_list = []
        self.error_list = []
        self.gan_list = []
        self.gan_list_name = []

    def train(self):
        nTrain = len(self.trainLoader.dataset)
        nProcessed = 0
        world_error = 0.0
        sum_loss = 0.0
        num = 0.0

        criterion = Modified_SmoothL1Loss().cuda()
        # criterion_mse = torch.nn.MSELoss()
        criterion_mse = torch.nn.MSELoss().cuda()
        # criterion = torch.nn.MSELoss()
        self.G.train()
        self.D.train()

        # for feature
        down = nn.AvgPool2d(args.pool_factor, stride=args.pool_factor)
        self.error_joint_list = []
        self.error_mean_list = []

        for batch_idx, data in enumerate(self.trainLoader):
            self.loss_list = []
            self.loss_list_name = []
            self.error_list = []
            self.gan_list = []
            self.gan_list_name = []
            error_list = []
            loss = 0.0
            num = num + 1

            img, joint3D, M, center, cube = data
            joint2D = self.xyz2uvd(joint3D.numpy(), center.numpy(), M.numpy(), cube.numpy(), self.trainLoader)
            joint2D = torch.from_numpy(joint2D).float().cuda()
            img, joint3D = img.cuda(), joint3D.cuda()
            batch_size = img.size(0)


            output, feature = self.G(img)
            output = output.view_as(joint2D)
            # loss_pos = criterion(output, joint3D)
            # joint3D_predict = self.uvd2xyz(output.detach().cpu().numpy(),center.numpy(),M.numpy(),cube.numpy(),self.trainLoader)
            # error = self.xyz2error(joint3D_predict, joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            loss_pos = criterion(output, joint3D)
            error = self.xyz2error(output.detach().cpu().numpy(),joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            self.loss_list_name.append('joint_xyz')
            loss_sup = loss_pos
            self.error_list.append(error)
            self.loss_list.append(loss_sup)

            # without discrimater
            if self.D_type =='NULL':
                self.G_opt.zero_grad()
                loss_sup.backward()
                self.G_opt.step()
            else:
                self.gan_trainer(img, down(img), point_GFM, output_GFM, joint_GFM, max_bbx_len, loss_sup, batch_idx)

            # update scheduler
            if self.scheduler_type =='SGDR':
                self.G_scheduler.step()
                self.D_scheduler.step()
            G_lr = self.G_scheduler.get_lr()[0]
            D_lr = self.D_scheduler.get_lr()[0]
            sum_loss += loss_sup.detach().cpu().numpy()
            world_error += self.error_list[-1]
            nProcessed += len(img)
            loss_info = ''
            error_info = ''
            gan_info = ''
            for index in range(len(self.loss_list)):
                loss_info += (self.loss_list_name[index] + ": " + "{:.6f}".format(self.loss_list[index]) + " ")
            for index in range(len(self.gan_list)):
                gan_info += (self.gan_list_name[index] + ": " + "{:.6f}".format(self.gan_list[index]) + " ")
            for index in range(len(self.error_list)):
                error_info += ("error" + str(index) + ": " + "{:.2f}".format(self.error_list[index]) + " ")
            print("Train Epoch: {:.2f} [{}/{} ({:.0f}%)] G_lr: {:.4f} D_lr: {:.4f} ".format(self.cur_epochs, nProcessed, nTrain, 100. * batch_idx * self.batch_size / nTrain, G_lr, D_lr) + loss_info + gan_info +error_info)
        logging.info('Epoch#%d: loss=%.4f, world_error:%.2f mm, lr = %f' % (self.cur_epochs, sum_loss / num, world_error / num, self.G_opt.param_groups[0]['lr']))
        # logging.info('Epoch#%d: loss=%.4f lr = %f' %(epoch, sum_loss/num, optimizer.param_groups[0]['lr']))

    def eval_model(self):
        print("Evaluating model on test set...")
        targets, predictions, crop_transforms, coms, data \
            = hape_eval.evaluate_model(self.G, self.testLoader, self.used_device)
        print("Computing metrics/Writing to files...")
        hape_eval.compute_and_output_results(targets, predictions, self.testLoader.dataset,
                                             args, self.model_dir)
        self.G.eval()
        error_sum = 0.0
        num = 0
        self.error_joint_list =[]
        self.error_mean_list = []
        for batch_idx, data in enumerate(self.testLoader):

            num = num + 1

            img, joint3D, M, center, cube = data
            joint2D = self.xyz2uvd(joint3D.numpy(), center.numpy(), M.numpy(), cube.numpy(), self.testLoader)
            joint2D = torch.from_numpy(joint2D).float().cuda()
            img, joint3D = img.cuda(), joint3D.cuda()
            batch_size = img.size(0)
            output, feature = self.G(img)
            output = output.view_as(joint3D)
            # joint3D_predict = self.uvd2xyz(output.detach().cpu().numpy(),center.numpy(),M.numpy(),cube.numpy(),self.testLoader)
            # error = self.xyz2error(joint3D_predict,joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            error = self.xyz2error(output.detach().cpu().numpy(),joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            error_sum += error
            print(error)
        print(error_sum/float(num))
        #


    def eval_gan(self, img, image_down, pcl_sample, vol, output, joint_label, batch_idx):
        if len(self.feature_name_list) != 1:
            input_real = []
            input_fake = []
            for feature_name in self.feature_name_list:
                if feature_name == 'heatmap' or feature_name == 'edt':
                    feature_label = self.generate_feature(img, joint_label, feature_name)
                    feature_output = self.generate_feature(img, output, feature_name)
                    input_real.append(torch.cat((image_down, feature_label), dim=1))
                    input_fake.append(torch.cat((image_down, feature_output), dim=1))

                    if feature_name == 'heatmap':
                        self.draw_feature('heatmap', input_real[0], input_fake[0], self.train_debug_dir, batch_idx)
                elif 'point' in feature_name:
                    feature_label = self.generate_feature(img, joint_label, feature_name)
                    feature_output = self.generate_feature(img, output, feature_name)
                    input_real.append(select_keypc(torch.cat((pcl_sample, feature_label), dim=2)))
                    input_fake.append(select_keypc(torch.cat((pcl_sample, feature_output), dim=2)))
                else:
                    input_real.append(self.generate_feature(img, joint_label, feature_name))
                    input_fake.append(self.generate_feature(img, output, feature_name))
        else:
            #single feature
            if 'joint' in self.feature_type:
                feature_label = joint_select.view(joint_select.size(0), -1)
                input_real = joint_select.view(joint_select.size(0), -1)
                input_fake = output
                feature_output = output
            elif '3D' in self.feature_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, output, self.feature_name_list[0])
                input_real = torch.cat((vol, feature_label), dim=1)
                input_fake = torch.cat((vol, feature_output), dim=1)
            elif 'point' in self.feature_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, output, self.feature_name_list[0])
                input_real = torch.cat((pcl_sample, feature_label), dim=2)
                input_fake = torch.cat((pcl_sample, feature_output), dim=2)
                if 'seg' not in self.D_type:
                    input_real = select_keypc(input_real)
                    input_fake = select_keypc(input_fake)
                self.draw_feature('point', input_real, input_fake, self.train_debug_dir, batch_idx)
            elif 'multiview' in self.feature_type:
                pcl_rot_label, feature_label = self.generate_feature(pcl_sample, joint_label, self.feature_name_list[0])
                pcl_rot_out, feature_output = self.generate_feature(pcl_sample, output, self.feature_name_list[0])
                input_real = torch.cat((pcl_rot_label, feature_label), dim=1)
                input_fake = torch.cat((pcl_rot_out, feature_output), dim=1)
                self.draw_feature('heatmap', input_real, input_fake, self.train_debug_dir, batch_idx)
            elif 'heatmap' in self.feature_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, output, self.feature_name_list[0])
                feature_random = self.generate_feature(img, torch.rand(output.size()).cuda(), self.feature_name_list[0])
                input_real = torch.cat((image_down, feature_label), dim=1)
                input_fake = torch.cat((image_down, feature_output), dim=1)
                input_random = torch.cat((image_down, feature_random), dim=1)
                # input_real = feature_label
                # input_fake = feature_output
                # self.draw_feature('heatmap', input_real, input_fake, self.test_debug_dir, batch_idx)
                self.draw_feature('heatmap', input_real, input_random, self.test_debug_dir, batch_idx)

        if self.adv_type == 'EB':
            # Discriminator
            loss_d_real = criterion_D(self.D(input_real), feature_label)
            loss_d_fake = criterion_D(self.D(input_fake.detach()), feature_output.detach())
            loss_d_random = criterion_D(self.D(input_random), feature_random)

            # Generator
            self.gan_list.append(loss_d_real)
            self.gan_list_name.append('loss_d_real')
            self.gan_list.append(loss_d_fake)
            self.gan_list_name.append('loss_d_fake')
            self.gan_list.append(loss_d_random)
            self.gan_list_name.append('loss_d_random')
        else:
            # real data
            d_out_real, dr1, dr2 = self.D(input_real)
            self.y.data.resize_(d_out_real.size(0)).fill_(1)
            # fake data
            if isinstance(input_fake, list):
                input_fake_detach = []
                for feature in input_fake:
                    input_fake_detach.append(feature.detach())
            else:
                input_fake_detach = input_fake.detach()

            d_out_fake, df1, df2 = self.D(input_fake_detach)

            if self.adv_type == 'wgan-gp':
                loss_real = - torch.mean(d_out_real)
                loss_fake = d_out_fake.mean()
            elif self.adv_type == 'hinge':
                loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            elif self.adv_type == 'Rahinge':
                loss_real = torch.mean(torch.nn.ReLU()(1.0 - (d_out_real - torch.mean(d_out_fake))))
                loss_fake = torch.mean(torch.nn.ReLU()(1.0 + (d_out_fake - torch.mean(d_out_real))))
            elif self.adv_type == 'RaLS':
                loss_real = torch.mean((d_out_real - torch.mean(d_out_fake) - self.y) ** 2)
                loss_fake = torch.mean((d_out_fake - torch.mean(d_out_real) + self.y) ** 2)
            self.loss_list_name.append('loss_real')
            self.loss_list.append(loss_real.detach().cpu())
            self.loss_list_name.append('loss_fake')
            self.loss_list.append(loss_fake.detach().cpu())
            self.loss_list_name.append('loss_diff')
            self.loss_list.append((loss_real - loss_fake).detach().cpu())

    def gan_trainer(self, img, image_down, pcl_sample, joint_predict, joint_label, max_bbx_len, loss_sup, batch_idx):
        #multi feature
        if len(self.feature_name_list) != 1:
            input_real = []
            input_fake = []
            for feature_name in self.feature_name_list:
                if feature_name == 'heatmap' or feature_name == 'edt':
                    feature_label = self.generate_feature(img, joint_label, feature_name)
                    feature_output = self.generate_feature(img, joint_predict, feature_name)
                    input_real.append(torch.cat((image_down, feature_label), dim=1))
                    input_fake.append(torch.cat((image_down, feature_output), dim=1))
                    if feature_name == 'heatmap':
                        self.draw_feature('heatmap', input_real[0], input_fake[0], self.train_debug_dir, batch_idx)
                elif 'point' in feature_name:
                    feature_label = self.generate_feature(img, joint_label, feature_name)
                    feature_output = self.generate_feature(img, joint_predict, feature_name)
                    input_real.append(select_keypc(torch.cat((pcl_sample, feature_label), dim=2)))
                    input_fake.append(select_keypc(torch.cat((pcl_sample, feature_output), dim=2)))
        else:
            # single feature
            if 'joint' == self.feature_type:
                feature_label = joint_select.view(joint_select.size(0), -1)
                input_real = joint_select.view(joint_select.size(0), -1)
                input_fake = joint_predict
                feature_output = joint_predict
            elif '3D' == self.feature_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, joint_predict, self.feature_name_list[0])
                input_real = torch.cat((vol, feature_label), dim=1)
                input_fake = torch.cat((vol, feature_output), dim=1)
            elif 'point_heatmap' == self.feature_type:
                feature_label = self.GFM_.joint2heatmap_pcl(joint_label, pcl_sample, max_bbx_len)
                feature_output = self.GFM_.joint2heatmap_pcl(joint_predict, pcl_sample, max_bbx_len)
                input_real = torch.cat((pcl_sample, feature_label), dim=1)
                input_fake = torch.cat((pcl_sample, feature_output), dim=1)
                self.draw_feature('point_heatmap', input_real, input_fake, self.train_debug_dir, batch_idx)
            elif 'point' == self.feature_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, joint_predict, self.feature_name_list[0])
                pcl_with_label = torch.cat((pcl_sample,torch.zeros(self.batch_size,1,self.sample_point).cuda()),dim=1)
                input_real = torch.cat((pcl_with_label, feature_label), dim=2)
                input_fake = torch.cat((pcl_with_label, feature_output), dim=2)
                self.draw_feature('point', input_real, input_fake, self.train_debug_dir, batch_idx)
            elif 'multiview' == self.feature_type:
                pcl_rot_label, feature_label = self.generate_feature(pcl_sample, joint_label, self.feature_name_list[0])
                pcl_rot_out, feature_output = self.generate_feature(pcl_sample, joint_predict,
                                                                    self.feature_name_list[0])
                input_real = torch.cat((pcl_rot_label, feature_label), dim=1)
                input_fake = torch.cat((pcl_rot_out, feature_output), dim=1)
                self.draw_feature('heatmap', input_real, input_fake, self.train_debug_dir, batch_idx)
            elif 'SA' == self.D_type:
                feature_label = self.generate_feature(img, joint_label, self.feature_name_list[0])
                feature_output = self.generate_feature(img, joint_predict, self.feature_name_list[0])
                input_real = torch.cat((image_down, feature_label), dim=1)
                input_fake = torch.cat((image_down, feature_output), dim=1)
                # input_real = feature_label
                # input_fake = feature_output
                self.draw_feature('heatmap', input_real, input_fake, self.train_debug_dir, batch_idx)

        if self.adv_type == 'EB':
            # Discriminator
            if self.D_type == 'point_reg':
                D_output_real = self.D(input_real).view(input_real.size(0),-1,3)
                D_output_fake = self.D(input_fake.detach()).view(input_real.size(0),-1,3)
                D_input_real = joint_label # for loss caculate
                D_input_fake = joint_predict.detach()
            else:
                D_output_real = self.D(input_real)
                D_output_fake = self.D(input_fake.detach())
                D_input_real = feature_label
                D_input_fake = feature_output.detach()

            loss_d_real = criterion_D(D_output_real, D_input_real)
            loss_d_fake = criterion_D(D_output_fake, D_input_fake)
            loss_d = loss_d_real - self.k * loss_d_fake
            self.D_opt.zero_grad()
            loss_d.backward()
            self.D_opt.step()

            # Generator
            if self.D_type == 'point_reg':
                D_output_fake = self.D(input_fake.detach()).view(input_real.size(0),-1,3)
                D_input_fake = joint_predict.detach()
            else:
                D_output_fake = self.D(input_fake.detach())
                D_input_fake = feature_output.detach()

            loss_g = criterion_D(D_output_fake, D_input_fake)
            loss_all = loss_sup + loss_g * args.coeff

            self.G_opt.zero_grad()
            loss_all.backward()
            self.G_opt.step()

            # update k
            loss_d_real_ = loss_d_real.detach().cpu().numpy()
            loss_d_fake_ = loss_d_fake.detach().cpu().numpy()
            balance = (self.gamma * loss_d_real_ - loss_d_fake_)/loss_d_fake_  # / FLAGS.lambda_G  # Is this dividing good? Original impl. has this
            self.k = self.k + self.lambda_k * balance
            self.k = float(min(1, max(0, self.k)))
            measure = loss_d_real + abs(balance)
            self.gan_list.append(self.k)
            self.gan_list_name.append('k')
            # self.gan_list.append(balance)
            # self.gan_list_name.append('balance')
            self.gan_list.append(loss_d_real)
            self.gan_list_name.append('loss_d_real_')
            self.gan_list.append(loss_d_fake)
            self.gan_list_name.append('loss_d_fake_')
        else:
            # real data
            d_out_real, dr1, dr2 = self.D(input_real)
            self.y.data.resize_(d_out_real.size(0)).fill_(1)
            # fake data
            if isinstance(input_fake, list):
                input_fake_detach = []
                for feature in input_fake:
                    input_fake_detach.append(feature.detach())
            else:
                input_fake_detach = input_fake.detach()

            d_out_fake, df1, df2 = self.D(input_fake_detach)

            if self.adv_type == 'wgan-gp':
                loss_d = - torch.mean(d_out_real) + d_out_fake.mean()
            elif self.adv_type == 'hinge':
                loss_d = torch.nn.ReLU()(1.0 - d_out_real).mean() + torch.nn.ReLU()(1.0 + d_out_fake).mean()
            elif self.adv_type == 'Rahinge':
                loss_d = (torch.mean(torch.nn.ReLU()(1.0 - (d_out_real - torch.mean(d_out_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (d_out_fake - torch.mean(d_out_real))))) / 2
            elif self.adv_type == 'RaLS':
                loss_d = (torch.mean((d_out_real - torch.mean(d_out_fake) - self.y) ** 2) + torch.mean((d_out_fake - torch.mean(d_out_real) + self.y) ** 2)) / 2
            self.D_opt.zero_grad()
            loss_d.backward()
            self.D_opt.step()


            if self.adv_type == 'wgan-gp':
                # Compute gradient penalty
                #input is list
                if isinstance(input_fake, list):
                    interpolated = []
                    for feature_index in range(len(input_real)):
                        alpha = torch.rand(input_real[feature_index].size(0), 1, 1, 1).cuda().expand_as(input_real[feature_index])
                        interpolated_temp = alpha * input_real[feature_index].detach() + (1 - alpha) * input_fake[feature_index].detach()
                        interpolated_temp.requires_grad = True
                        interpolated.append(interpolated_temp)
                else:
                    alpha = torch.rand(input_real.size(0), 1, 1, 1).cuda().expand_as(input_real)
                    interpolated = alpha * input_real.detach() + (1 - alpha) * input_fake.detach()
                    interpolated.requires_grad = True

                out, _, _ = self.D(interpolated)


                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                loss_d = self.lambda_gp * d_loss_gp

                self.D_opt.zero_grad()
                loss_d.backward()
                self.D_opt.step()

            # Compute loss with fake images
            # get real data
            d_out_real, dr1, dr2 = self.D(input_real)
            self.y.data.resize_(d_out_real.size(0)).fill_(1)

            g_out_fake, _, _ = self.D(input_fake)  # batch x n

            if self.adv_type == 'wgan-gp':
                loss_g = - g_out_fake.mean()
            elif self.adv_type == 'hinge':
                loss_g = - g_out_fake.mean()
            elif self.adv_type == 'Rahinge':
                loss_g = (torch.mean(torch.nn.ReLU()(1.0 + (d_out_real - torch.mean(g_out_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (g_out_fake - torch.mean(d_out_real))))) / 2
            elif self.adv_type == 'RaLS':
                loss_g = (torch.mean((d_out_real - torch.mean(g_out_fake) + self.y) ** 2) + torch.mean((g_out_fake - torch.mean(d_out_real) - self.y) ** 2)) / 2

            loss_all = self.coeff * loss_g + loss_sup
            # loss_all = self.coeff * loss_g
            self.G_opt.zero_grad()
            loss_all.backward()
            self.G_opt.step()

            self.loss_list_name.append('loss_g')
            self.loss_list.append(loss_g.detach().cpu())
            self.loss_list_name.append('loss_d')
            self.loss_list.append(loss_d.detach().cpu())

    def generate_feature(self, img, joint, feature_type):
        joint = joint.view(joint.size(0), -1, 3)
        if feature_type == 'heatmap':
            heatmap2d = self.GFM_.joint2heatmap2d(joint, isFlip=False)
            depth = self.GFM_.depth2map(joint[:, :, 2])
            feature = heatmap2d * (depth+1)/2
            if self.pool_factor / 2 > 1:
                down = nn.MaxPool2d(self.pool_factor / 2, stride=self.pool_factor / 2)
                feature = down(feature)
        elif feature_type == 'heatmap3D':
            feature = self.GFM_.joint2heatmap3d(joint, isFlip=False)
        elif feature_type == 'heatmap_multiview':
            pcl_sample = img
            rot = np.random.rand(joint.size(0), 3) * np.pi
            pcl_sample_rot = self.GFM_.rotation_points(pcl_sample, rot)
            joint_rot = self.GFM_.rotation_points(joint, rot)

            pcl_img = self.GFM_.pcl2img(pcl_sample_rot, self.input_size/self.pool_factor)
            heatmap2d = self.GFM_.joint2heatmap2d(joint_rot)
            depth = self.GFM_.depth2map(joint_rot[:, :, 2])
            feature = heatmap2d * depth
            if self.pool_factor / 2 > 1:
                down = nn.MaxPool2d(self.pool_factor / 2, stride=self.pool_factor / 2)
                feature = down(feature)
            return pcl_img, feature
        elif feature_type == 'point':
            feature = self.GFM_.joint2pc(joint, sample_point=self.sample_point)
        elif feature_type == 'GD':
            feature = self.GFM_.joint2GD(joint)
        elif feature_type == 'edt':
            edtmap = self.GFM_.imgs2edt2(img, joint, img.size(-1) / self.pool_factor)
            # distmap = self.GFM_.imgs2dist(img, joint, img.size(-1) / pool_factor)
            distmap = self.GFM_.imgs2dist_tensor(img, joint, img.size(-1) / self.pool_factor)
            feature = edtmap * distmap
        else:
            print("Invalid feature type, exiting...")
            exit()
        return feature

    def draw_feature(self, feature_type, feature_label, feature_output, draw_dir, batch_idx):
        if feature_type == 'heatmap':
            label = feature_label[0].cpu().numpy()
            output = feature_output[0].detach().cpu().numpy()
            heatmap_draw = np.zeros((feature_output.size(-1), feature_output.size(-1), 1))
            heatmap_draw[:, :, 0] = ((output[0]+1)/4 + (output[1:].sum(0)/2)) * 255.0
            img_name_1 = draw_dir + '/output_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name_1, heatmap_draw)
            heatmap_draw[:, :, 0] = ((label[0]+1)/4 + (label[1:].sum(0)/2)) * 255.0
            img_name_1 = draw_dir + '/label_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name_1, heatmap_draw)
        elif feature_type == 'edt':
            heatmap_draw = np.zeros((feature_output.size(-1), feature_output.size(-1), 1))
            heatmap_draw[:, :, 0] = (feature_output[0, 0].detach().cpu().numpy()) * 255.0
            img_name_1 = draw_dir + '/output_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name_1, heatmap_draw)
            heatmap_draw[:, :, 0] = (feature_label[0, 0].cpu().numpy()) * 255.0
            img_name_1 = draw_dir + '/label_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name_1, heatmap_draw)
        elif feature_type == 'point':
            img_size = self.input_size
            pcl = feature_output.detach().cpu().numpy().transpose(0, 2, 1)
            img_pcl = np.ones([img_size, img_size, 1]) * -1
            index_x = np.clip(np.floor((pcl[0, :, 0] + 1) / 2 * img_size), 0, img_size - 1).astype('int')
            index_y = np.clip(np.floor((1 - pcl[0, :, 1]) / 2 * img_size), 0, img_size - 1).astype('int')
            img_pcl[index_y, index_x, 0] = 255.0
            pcl = feature_label.cpu().numpy().transpose(0, 2, 1)
            img_label = np.ones([img_size, img_size, 1]) * -1
            index_x = np.clip(np.floor((pcl[0, :, 0] + 1) / 2 * img_size), 0, img_size - 1).astype('int')
            index_y = np.clip(np.floor((1 - pcl[0, :, 1]) / 2 * img_size), 0, img_size - 1).astype('int')
            img_label[index_y, index_x, 0] = 255.0
            img_name = draw_dir + '/output_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name, img_pcl)
            img_name = draw_dir + '/label_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name, img_label)
        elif feature_type == 'depth':
            heatmap_draw = np.zeros((feature_output.size(-1), feature_output.size(-1), 1))
            heatmap_draw[:, :, 0] = ((feature_output[0, 0].detach().cpu().numpy())+1)/2 * 255.0
            img_name = draw_dir + '/output_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name, heatmap_draw)
            heatmap_draw[:, :, 0] = ((feature_label[0, 0].cpu().numpy())+1)/2 * 255.0
            img_name = draw_dir + '/label_' + str(batch_idx) + '.png'
            cv2.imwrite(img_name, heatmap_draw)
        elif feature_type == 'point_heatmap':
            for idx_joint in range(self.joint_num):
                img = draw_depth_heatmap(self.dataset, feature_label[0,0:3,:].cpu().numpy(), feature_label[0,3:,:].cpu().numpy(), idx_joint)
                img_name = draw_dir + '/label_' + str(batch_idx)+'_'+str(idx_joint) + '.png'
                cv2.imwrite(img_name, img)
                img = draw_depth_heatmap(self.dataset, feature_output[0,0:3,:].detach().cpu().numpy(), feature_output[0,3:,:].detach().cpu().numpy(), idx_joint)
                img_name = draw_dir + '/output_' + str(batch_idx)+'_'+str(idx_joint)  + '.png'
                cv2.imwrite(img_name, img)

    def xyz2error(self, output, joint, center, cube_size):
        with torch.no_grad():
            batchsize, joint_num, _ = output.shape
            center = np.tile(center.reshape(batchsize, 1, 3), [1, joint_num, 1])
            cube_size = np.tile(cube_size.reshape(batchsize, 1, 3), [1, joint_num, 1])

            output = output * cube_size / 2 + center
            joint_xyz_label = joint * cube_size / 2 + center

            temp = (output - joint_xyz_label) * (output - joint_xyz_label)
            error = np.sqrt(np.sum(temp, 2))
            self.error_joint_list.append(error)
            self.error_mean_list.append(np.mean(error, axis=1))
            error = np.mean(error)

        return error

    def uvd2error(self, output, joint, center, M, cube_size, loader, file=None):
        with torch.no_grad():
            joint_xyz, joint_uvd = loader.jointsuvd2xyz(output, center, M, cube_size)
            batchsize, joint_num, _ = output.shape
            center = np.tile(center.reshape(batchsize, 1, 3), [1, joint_num, 1])
            cube_size = np.tile(cube_size.reshape(batchsize, 1, 3), [1, joint_num, 1])

            joint_xyz_label = joint * cube_size / 2 + center

            temp = (joint_xyz - joint_xyz_label) * (joint_xyz - joint_xyz_label)
            error_list = np.sqrt(np.sum(temp, 2))
            self.error_joint_list.append(error_list)
            self.error_mean_list.append(np.mean(error_list, axis=1))
            error = np.mean(error_list)
            if not file == None:
                # np.savetxt(file, joint_xyz.reshape([batchsize, joint_num*3]), fmt='%.2f')
                np.savetxt(file, joint_uvd.reshape([batchsize, joint_num * 3]), fmt='%.3f')
        return error

    def xyz2uvd(self, joint, center, M, cube_size, loader):
        joint3d = joint.copy()
        targets = loader.dataset.denormalize_joint_pos(joint3d, cube_size)
        targets = targets + center.reshape(center.shape[0], 1, center.shape[1])
        joint2d = loader.dataset.points3DToImg(targets.reshape(-1, 3))
        joint2d = joint2d.reshape(-1,self.joint_num,3)
        joint2d[:,:,2] = 1
        joint2d = np.matmul(np.tile(M.reshape(-1,1,3,3),(1,self.joint_num,1,1)),joint2d.reshape(-1,self.joint_num,3,1)).squeeze()
        joint2d[:,:,0:2] = joint2d[:, :, 0:2] / (128.0 / 2) - 1
        joint2d[:,:,2] = joint3d[:, :, 2]
        return joint2d

    def uvd2xyz(self, joint, center, M, cube_size, loader):
        joint2d = joint.copy()
        joint2d[:, :, 0:2] = (joint2d[:,:, 0:2] + 1)  * (128.0 / 2)
        joint2d[:, :, 2]  = 1
        joint2d[:, :, 0:2] = np.matmul(inv(np.tile(M.reshape(-1,1,3,3),(1,self.joint_num,1,1))),joint2d.reshape(-1,self.joint_num,3,1)).squeeze()[:,:,0:2]
        joint2d[:, :, 2] = joint[:,:, 2] *  (cube_size[:,2].reshape(cube_size.shape[0], 1) / 2)
        joint2d[:, :, 2] = joint2d[:, :, 2] + center[:,2].reshape(-1,1)
        joint3d = loader.dataset.pointsImgTo3D(joint2d.reshape(-1, 3)).reshape(-1,self.joint_num,3)
        joint3d = (joint3d -  center.reshape(center.shape[0], 1, 3)) / (cube_size/2).reshape(-1,1,3)
        return joint3d

def select_keypc(point):
    device = point.device
    point = point.permute(0, 2, 1)
    batch_size, point_num, _ = point.size()
    positive = torch.ones([batch_size, point_num/2, 1]).to(device)
    negative = torch.ones([batch_size, point_num/2, 1]).to(device)*-1
    # return torch.cat((point[:, 1024:, :], positive), dim=-1)
    label = torch.cat((positive,negative),dim=1)
    point = torch.cat((point,label),dim=-1)
    sample_img_1 = torch.randperm(1024)
    sample_joint_1 = torch.randperm(1024) + 1024

    point[:, 0:1024, :] = point[:, sample_img_1, :]
    point[:, 1024:, :] = point[:, sample_joint_1, :]

    point_list = torch.split(point, point_num/32, dim=1)#min block is 64
    temp_point = torch.cat((point_list[0], point_list[16]), dim=1)
    for index in range(1, 8):
        temp_point = torch.cat((temp_point, point_list[index], point_list[16+index]), dim=1)
    return temp_point.permute(0, 2, 1)


def heatmap2error(heatmap, depth, joint_xyz_label, test_data, batchid, heatmapSz):
    batchsize, joint_num, _, _ = heatmap.shape

    center = test_data.test_data_com[batchsize * batchid: batchsize * (batchid + 1)]
    center_uvd = np.zeros([batchsize, 3])
    for n in range(batchsize):
        center_uvd[n] = test_data.di.joint3DToImg(center[n])
    cube_size = test_data.test_data_cube[batchsize * batchid: batchsize * (batchid + 1), 0]
    M = test_data.test_data_M[batchsize * batchid: batchsize * (batchid + 1)]
    with torch.no_grad():
        joint_uvd = loader.heatmap2joints(heatmap, depth, center, cube_size, heatmapSz, M)
        joint_xyz = np.zeros_like(joint_uvd)
        for n in range(batchsize):
            joint_xyz[n] = test_data.di.jointsImgTo3D(joint_uvd[n])

        center = np.tile(center.reshape(batchsize, 1, 3), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, 1), [1, joint_num, 3])
        joint_xyz_label = joint_xyz_label * cube_size / 2 + center
        temp = (joint_xyz - joint_xyz_label) * (joint_xyz - joint_xyz_label)
        error = np.mean(np.sqrt(np.sum(temp, 2)))
    return error


def mse_loss(input, target):
    return torch.sum((input - target) * (input - target)) / input.data.nelement()


def criterion_D(pred, gt):
    l = (pred - gt) ** 2
    l = l.mean()
    return l


def draw_poses_deep_piror(test_data, joint_xyz, subname):
    batchsize, jointnum, _ = joint_xyz.shape
    # because predict data's batchsize no match dataset's batchsize
    pd3D = joint_xyz[:, calculate2deeppiror, :] * np.tile(test_data.test_data_cube.reshape(-1, 1, 3), (1, jointnum, 1))[
                                                  :batchsize] / 2 + np.tile(test_data.test_data_com.reshape(-1, 1, 3),
                                                                            (1, jointnum, 1))[:batchsize]
    if test_data.alljoint_numoints:
        gt3D = test_data.test_gt3D * np.tile(test_data.test_data_cube.reshape(-1, 1, 3),
                                             (1, test_data.joint_num, 1)) / 2 + np.tile(
            test_data.test_data_com.reshape(-1, 1, 3), (1, test_data.joint_num, 1))
    else:
        gt3D = test_data.test_gt3D * np.tile(test_data.test_data_cube.reshape(-1, 1, 3),
                                             (1, test_data.joint_num, 1)) / 2 + np.tile(
            test_data.test_data_com.reshape(-1, 1, 3), (1, test_data.joint_num, 1))
    gt3D = gt3D[:batchsize][:, calculate, :][:, calculate2deeppiror, :]
    hpe = NYUHandposeEvaluation(gt3D, pd3D)
    hpe.subfolder += '/' + subname + '/'
    # hpe.plotEvaluation('debug_plot', methodName='Our regr')
    test_seq_num = len(test_data.testSeqs)
    ind = 0
    for a in range(test_seq_num):
        for i in test_data.testSeqs[a].data:
            jtI = transformPoints2D(test_data.di.joints3DToImg(pd3D[ind]), i.T)
            hpe.plotResult(i.dpt, i.gtcrop[restrictedjoint_numointsEval, :], jtI, "{}".format(ind), niceColors=True)
            ind += 1
            if ind >= batchsize:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--root_dir', type=str, default='/data/users/pfren/data/dataset/hand')

    parser.add_argument('--load_epoch', default=68, type=int)
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_max', type=int, default=80)
    parser.add_argument('--G_lr', type=float, default=3.3e-4)
    parser.add_argument('--D_lr', type=float, default=1e-3)#SA=0.0004, point=1e-3, V2V=2.5e-4, point_plus=1e-3
    parser.add_argument('--G_step_size', type=int, default=20)
    parser.add_argument('--D_step_size', type=int, default=20)
    parser.add_argument('--finetune_dir', default='', type=str)# resnet18/latest_G80.pth 'resnet18/P0/latest_G60.pth'resnet18/P0/latest_G80.pth
    parser.add_argument('--G_opt_type', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--D_opt_type', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--scheduler_type', type=str, default='warm-up', choices=('SGDR', 'step','warm-up'))

    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--dataset', default='nyu', type=str, choices=['nyu', 'icvl', 'msra','itop'])
    parser.add_argument('--view', default='side', type=str, choices=['side', 'top'])
    parser.add_argument('--test_id', default=0, type=int) # for msra
    parser.add_argument('--joint_num', default=14, type=int)  # nyu 14 icvl 16 msra 21 itop 15
    parser.add_argument('--model_para', default=47, type=int)  # nyu 69/47 icvl 51 msra 62

    parser.add_argument('--model_save', default='resnet_mur_Gsgd0.3_stepbatch128', type=str)#'point_plus_fintune_uvd2xyz_Glr0.1_Dlr1e-3/P0
    parser.add_argument('--G_type', default='resnet_mur', type=str)
    parser.add_argument('--D_type', default='NULL', type=str, choices=['SA', 'SA_M', 'SA3D', 'HG', 'point_plus','point_seg','point_reg','linear','NULL'])
    parser.add_argument('--adv_type', default='EB', type=str, choices=['EB', 'wgan-gp', 'hinge', 'Rahinge', 'RaLS'])
    parser.add_argument('--pool_factor', default=2, type=int)
    parser.add_argument('--ball_radius', default=0.04, type=int) #0.04,0.08
    parser.add_argument('--ball_radius2', default=0.08, type=int)#0.12,0.16
    parser.add_argument('--sample_point', default=1024, type=int)
    parser.add_argument('--feature_type', default='point_heatmap', type=str, choices=['heatmap', 'edt', 'GD', 'heatmap3D','point_heatmap', 'point', 'heatmap_multiview'])
    parser.add_argument('--train_debug_dir', default='./debug/point_heatmap', type=str)
    parser.add_argument('--test_debug_dir', default='./debug/heatmap_nyu', type=str)

    parser.add_argument('--k', default=0, type=float)
    parser.add_argument('--lambda_k', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)# the G is better than D
    parser.add_argument('--coeff', default=0.0001, type=float)
    parser.add_argument('--lambda_gp', default=10, type=float)


    args_my = parser.parse_args()
    # %% Set configuration
    from config.config import args  # general parameters

    args = parse_arguments_generic(args)  # command-line parameters
    from config.config_data_nyu import args_data  # dataset specific parameters

    args.__dict__ = dict(args.__dict__.items() + args_data.__dict__.items() + args_my.__dict__.items() )  # merge into single object

    trainer = Trainer(args,args)
    # test
    if trainer.phase == "test":
        trainer.G.load_state_dict(torch.load(os.path.join(trainer.model_dir + '/latest_G' + str(trainer.load_epoch) + '.pth'),
                                          map_location=lambda storage, loc: storage))
        trainer.eval_model()
        exit()
    iter_num_SGDR = int(float(len(trainer.trainData))/trainer.batch_size)
    epoch_update = 1
    update_size = 1
    for epoch in range(trainer.load_epoch + 1, trainer.epoch_max + 1):
        print("start epoch : " + str(epoch))
        if trainer.scheduler_type == 'step':
            if trainer.G_opt_type == 'sgd' or trainer.G_opt_type == 'adam':
                trainer.G_scheduler.step(epoch - 1)
            if trainer.D_opt_type == 'sgd'or trainer.D_opt_type == 'adam':
                trainer.D_scheduler.step(epoch - 1)
        elif trainer.scheduler_type == 'SGDR' and epoch == epoch_update:
            print('Reset scheduler')
            iter_num_SGDR = iter_num_SGDR * 2
            update_size = update_size * 2
            epoch_update = epoch_update + update_size
            trainer.G_opt = optim.SGD(trainer.G.parameters(), lr=trainer.G_lr, momentum=0.9, weight_decay=1e-4)
            trainer.D_opt = optim.SGD(trainer.D.parameters(), lr=trainer.D_lr, momentum=0.9, weight_decay=1e-4)
            trainer.G_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.G_opt, iter_num_SGDR)
            trainer.D_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.D_opt, iter_num_SGDR)
        trainer.train()
        torch.save(trainer.G.state_dict(), os.path.join(trainer.model_dir, 'latest_G' + str(epoch) + '.pth'))
        if trainer.D_type != 'NULL':
            torch.save(trainer.D.state_dict(), os.path.join(trainer.model_dir, 'latest_D' + str(epoch) + '.pth'))
        # trainer.eval_model()
        trainer.cur_epochs = trainer.cur_epochs + 1


