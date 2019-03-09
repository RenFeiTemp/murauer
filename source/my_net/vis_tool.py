import numpy as np
import torch
from torch.autograd import Variable
import cv2
from enum import Enum
calculate  = [0,3,5,8,10,13,15,18,24,25,26,28,29,30]

def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu_full':
        return 240.99, 240.96, 160, 120
        # return 588.03, 587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'hands17':
        return 475.065948, 475.065857, 315.944855, 245.287079
    elif dataset == 'xtion2':
        return 535.4, 539.2, 320.1, 247.6
    elif dataset == 'itop':
        return 285.71, 285.71, 160.0, 120.0


def get_joint_num(dataset):
    joint_num_dict = {'nyu': 14,'nyu_full': 23, 'icvl': 16, 'msra': 21, 'hands17': 21, 'itop': 15}
    return joint_num_dict[dataset]


def pixel2world(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, dataset):
    fx,fy,ux,uy = get_param(dataset)
    x[:, :, 0] = x[:, :, 0] * fx/x[:, :, 2] + ux
    x[:, :, 1] = uy - x[:, :, 1] * fy / x[:, :, 2]
    return x

def jointImgTo3D(uvd, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(uvd, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (uvd[0] - fu) * uvd[2] / fx
        ret[1] = (uvd[1] - fv) * uvd[2] / fy
        ret[2] = uvd[2]
    else:
        ret[:, 0] = (uvd[:,0] - fu) * uvd[:, 2] / fx
        ret[:, 1] = (uvd[:,1] - fv) * uvd[:, 2] / fy
        ret[:, 2] = uvd[:,2]
    return ret


def joint3DToImg(xyz, paras):
    fx, fy, fu, fv = paras
    ret = np.zeros_like(xyz, np.float32)
    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    else:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    return ret

def save_result_img(index, root_dir,pic_dir, pose):
    img = cv2.imread(root_dir + '/convert/' + '{}.jpg'.format(index), 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    draw_pose(img, pose)
    cv2.imwrite(pic_dir+'/' + str(index) + ".png", img)


def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'nyu_full':
        return [(20,3),(3,2),(2,1),(1,0),(20,7),(7,6),(6,5),(5,4),(20,11),(11,10),(10,9),(9,8),(20,15),(15,14),(14,13),(13,12),(20,19),(19,18),(18,17),(17,16),
               (20,21),(20,22)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'hands17':
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]
    elif dataset == 'itop':
        return [(0, 1),
                (1, 2), (2, 4), (4, 6),
                (1, 3), (3, 5), (5, 7),
                (1, 8),
                (8, 9), (9, 11), (11, 13),
                (8, 10), (10, 12), (12, 14)]

class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)

def get_sketch_color(dataset):
    if dataset == 'icvl':
        return [Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.BLUE, Color.GREEN,
                Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN,Color.GREEN,Color.GREEN,Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,  Color.PURPLE, Color.PURPLE,Color.PURPLE,Color.PURPLE,
                Color.YELLOW,Color.YELLOW,Color.YELLOW,Color.YELLOW,
                Color.BLUE, Color.BLUE,  Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]
    elif dataset == 'itop':
        return [Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              ]

def get_joint_color(dataset):
    if dataset == 'icvl':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN, Color.GREEN,Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]
    elif dataset == 'itop':
        return  [Color.RED,Color.BROWN,
                 Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE,
                 Color.CYAN,
                 Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE,Color.YELLOW,Color.PURPLE]

def draw_pose(dataset, img, pose):
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    idx = 0
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, colors_joint[idx].value, -1)
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 1)
        idx = idx + 1
    return img

def draw_depth_heatmap(dataset,pcl,heatmap,joint_id):
    fx, fy, ux, uy = get_param(dataset)
    pcl = pcl.transpose(1, 0)
    # pcl = joint3DToImg(pcl,(fx, fy, ux, uy))
    pcl = (pcl + 1) * 64
    sample_num = pcl.shape[0]
    img = np.ones((128, 128), dtype=np.uint8)*255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors_joint = get_joint_color(dataset)
    for idx in range(sample_num):
        r = int(colors_joint[joint_id].value[0] * heatmap[joint_id,idx])
        b = int(colors_joint[joint_id].value[1] * heatmap[joint_id,idx])
        g = int(colors_joint[joint_id].value[2] * heatmap[joint_id,idx])
        cv2.circle(img, (int(pcl[idx,0]), int(pcl[idx,1])), 1,(r,g,b) , -1)
    return img


