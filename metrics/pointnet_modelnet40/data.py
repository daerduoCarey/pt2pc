import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        print(h5_name)
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def rotate_point_cloud(pc):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    jittered_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_pc += pc
    return jittered_pc


def nocslize(pc):
    # z-up
    new_pc = np.array(pc, dtype=np.float32)
    new_pc[:, 2] = pc[:, 1]
    new_pc[:, 1] = -pc[:, 2]

    # nocs
    xyz_min = np.min(new_pc, axis=0)
    xyz_max = np.max(new_pc, axis=0)
    xyz_center = (xyz_min + xyz_max) / 2
    scale = np.sqrt(((xyz_max - xyz_min) ** 2).sum())
    new_pc[:, 0] -= xyz_center[0]
    new_pc[:, 1] -= xyz_center[1]
    new_pc[:, 2] -= xyz_center[2]
    new_pc /= scale

    return new_pc


class ModelNet40(Dataset):
    def __init__(self, partition='train'):
        self.data, self.label = load_data(partition)
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        pointcloud = rotate_point_cloud(pointcloud)
        pointcloud = jitter_point_cloud(pointcloud)
        pointcloud = nocslize(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

