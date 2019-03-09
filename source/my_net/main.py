import os
import sys
import math
import logging
import argparse
import numpy as np
from numpy.linalg import inv
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import  transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


sys.path.append('..')
from util.argparse_helper import parse_arguments_generic
from data.NyuHandPoseDataset import NyuHandPoseMultiViewDataset,NyuHandPoseDataset
from data.basetypes import LoaderMode, DatasetType
from util.transformations import transformPoint2D
import eval.handpose_evaluation as hape_eval
from resnet_mur import resnet50,resnet18

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
        self.batch_size = config.batch_size
        self.load_epoch = config.load_epoch
        self.epoch_max = config.epoch_max
        self.phase = config.phase
        self.gpu_id = config.gpu_id
        self.used_device = torch.device("cuda:{}".format(self.gpu_id))
        self.dataset = config.dataset
        self.test_id = config.test_id
        self.model_save = config.model_save
        self.G_type = config.G_type

        self.view = config.view
        self.input_size = config.input_size
        self.joint_num = config.joint_num


        self.G_lr = config.G_lr
        self.G_step_size = config.G_step_size
        self.G_opt_type = config.G_opt_type
        self.scheduler_type = config.scheduler_type
        # warm-up
        self.lr_lambda = lambda epoch: (0.33 ** max(0, 2 - epoch // 2)) if epoch < 4 else np.exp(-0.04 * epoch)


        self.data_rt = self.root_dir + "/" + self.dataset
        if self.model_save == '':
                self.model_save = self.G_type + '_' + 'G' + str(self.G_opt_type) + str(self.G_lr) + "_" + self.scheduler_type + 'batch' + str(self.batch_size)

        self.model_dir = './model/' + self.dataset + '/' + self.model_save
        self.cur_epochs = self.load_epoch

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.model_dir + '/img'):
            os.mkdir(self.model_dir + '/img')

        # use GPU
        self.cuda = torch.cuda.is_available()
        gpu_num = 1
        torch.cuda.set_device(self.gpu_id)
        cudnn.benchmark = True

        # set network
        if 'basic' in self.G_type:
            self.G = resnet18(pretrained=False, num_classes = 3 * self.joint_num)
        elif 'resnet_mur' in self.G_type:
            self.G = resnet50(num_classes = self.joint_num*3)

        self.G.apply(weights_init_resnet)
        if gpu_num > 1:
            self.G = nn.DataParallel(self.G).cuda()
        else:
            self.G = self.G.cuda()

        # load data
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')

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
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.testLoader = DataLoader(self.testData, batch_size=self.batch_size, shuffle=False, num_workers=8)

        else:
            if self.dataset == 'nyu':
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

            self.testLoader = DataLoader(self.testData, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if self.G_opt_type == 'sgd':
            self.G_opt = optim.SGD(self.G.parameters(), lr=self.G_lr, momentum=0.9, weight_decay=1e-4)
        elif self.G_opt_type == 'adam':
            self.G_opt = optim.Adam(self.G.parameters(), lr=self.G_lr)
        elif self.G_opt_type == 'rmsprop':
            self.G_opt = optim.RMSprop(self.G.parameters(), lr=self.G_lr)

        if self.scheduler_type == 'step':
            self.G_scheduler = lr_scheduler.StepLR(self.G_opt, step_size=self.G_step_size, gamma=0.1)
        elif self.scheduler_type =='SGDR':
            self.G_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.G_opt, int(float(len(self.trainData))/self.batch_size))
        elif self.scheduler_type == 'warm-up':
            self.G_scheduler = optim.lr_scheduler.LambdaLR(self.G_opt, lr_lambda=self.lr_lambda)

        # load model
        if self.load_epoch != 0:
            self.G.load_state_dict(torch.load(os.path.join(self.model_dir + '/latest_G' + str(self.load_epoch) + '.pth'),
                                              map_location=lambda storage, loc: storage))
        self.loss_list_name = []
        self.loss_list = []
        self.error_list = []

    def train(self):
        nTrain = len(self.trainLoader.dataset)
        nProcessed = 0
        world_error = 0.0
        sum_loss = 0.0
        num = 0.0

        criterion = Modified_SmoothL1Loss().cuda()
        self.G.train()
        for batch_idx, data in enumerate(self.trainLoader):
            self.loss_list = []
            self.loss_list_name = []
            self.error_list = []

            num = num + 1

            img, joint3D, M, center, cube = data
            img, joint3D = img.cuda(), joint3D.cuda()

            output, feature = self.G(img)
            output = output.view_as(joint3D)
            loss_pos = criterion(output, joint3D)
            error = self.xyz2error(output.detach().cpu().numpy(),joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            self.loss_list_name.append('joint_xyz')
            loss_sup = loss_pos
            self.error_list.append(error)
            self.loss_list.append(loss_sup)

            self.G_opt.zero_grad()
            loss_sup.backward()
            self.G_opt.step()

            # update scheduler
            if self.scheduler_type =='SGDR':
                self.G_scheduler.step()
            G_lr = self.G_scheduler.get_lr()[0]
            sum_loss += loss_sup.detach().cpu().numpy()
            world_error += self.error_list[-1]
            nProcessed += len(img)
            loss_info = ''
            error_info = ''
            for index in range(len(self.loss_list)):
                loss_info += (self.loss_list_name[index] + ": " + "{:.6f}".format(self.loss_list[index]) + " ")
            for index in range(len(self.error_list)):
                error_info += ("error" + str(index) + ": " + "{:.2f}".format(self.error_list[index]) + " ")
            print("Train Epoch: {:.2f} [{}/{} ({:.0f}%)] G_lr: {:.4f} ".format(self.cur_epochs, nProcessed, nTrain, 100. * batch_idx * self.batch_size / nTrain, G_lr) + loss_info +error_info)
        logging.info('Epoch#%d: loss=%.4f, world_error:%.2f mm, lr = %f' % (self.cur_epochs, sum_loss / num, world_error / num, self.G_opt.param_groups[0]['lr']))

    def eval_model(self):
        # print("Evaluating model on test set...")
        # targets, predictions, crop_transforms, coms, data \
        #     = hape_eval.evaluate_model(self.G, self.testLoader, self.used_device)
        # print("Computing metrics/Writing to files...")
        # hape_eval.compute_and_output_results(targets, predictions, self.testLoader.dataset,
        #                                      args, self.model_dir)
        self.G.eval()
        error_sum = 0.0
        num = 0
        for batch_idx, data in enumerate(self.testLoader):
            num = num + 1
            img, joint3D, M, center, cube = data
            img, joint3D = img.cuda(), joint3D.cuda()
            output, feature = self.G(img)
            output = output.view_as(joint3D)
            error = self.xyz2error(output.detach().cpu().numpy(),joint3D.cpu().numpy(), center.numpy(), cube.numpy())
            error_sum += error
            print(error)
        print(error_sum/float(num))

    def xyz2error(self, output, joint, center, cube_size):
        with torch.no_grad():
            batchsize, joint_num, _ = output.shape
            center = np.tile(center.reshape(batchsize, 1, 3), [1, joint_num, 1])
            cube_size = np.tile(cube_size.reshape(batchsize, 1, 3), [1, joint_num, 1])

            output = output * cube_size / 2 + center
            joint_xyz_label = joint * cube_size / 2 + center

            temp = (output - joint_xyz_label) * (output - joint_xyz_label)
            error = np.sqrt(np.sum(temp, 2))
            error = np.mean(error)

        return error


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--root_dir', type=str, default='/data/users/pfren/data/dataset/hand')

    parser.add_argument('--load_epoch', default=0, type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch_max', type=int, default=80)
    parser.add_argument('--G_lr', type=float, default=0.3)
    parser.add_argument('--G_step_size', type=int, default=20)
    parser.add_argument('--G_opt_type', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--scheduler_type', type=str, default='step', choices=('SGDR', 'step','warm-up'))

    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--dataset', default='nyu', type=str, choices=['nyu', 'icvl', 'msra','itop'])
    parser.add_argument('--view', default='side', type=str, choices=['side', 'top'])
    parser.add_argument('--test_id', default=0, type=int) # for msra
    parser.add_argument('--joint_num', default=14, type=int)  # nyu 14 icvl 16 msra 21 itop 15
    parser.add_argument('--model_para', default=47, type=int)  # nyu 69/47 icvl 51 msra 62

    parser.add_argument('--model_save', default='', type=str)
    parser.add_argument('--G_type', default='basic', type=str)

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
        elif trainer.scheduler_type == 'SGDR' and epoch == epoch_update:
            print('Reset scheduler')
            iter_num_SGDR = iter_num_SGDR * 2
            update_size = update_size * 2
            epoch_update = epoch_update + update_size
            trainer.G_opt = optim.SGD(trainer.G.parameters(), lr=trainer.G_lr, momentum=0.9, weight_decay=1e-4)
            trainer.G_scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.G_opt, iter_num_SGDR)
        trainer.train()
        torch.save(trainer.G.state_dict(), os.path.join(trainer.model_dir, 'latest_G' + str(epoch) + '.pth'))
        trainer.eval_model()
        trainer.cur_epochs = trainer.cur_epochs + 1


