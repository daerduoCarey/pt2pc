import os
import sys
import torch
import torch.nn as nn
import numpy as np
from progressbar import ProgressBar
from model import PointNet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

category = sys.argv[1]
split = sys.argv[2]
print(category, split)

device = torch.device('cuda:0')

model = PointNet().to(device)
path = "./pointnet_modelnet40/checkpoints/pointnet_max_pc_2048_emb_1024/models/model.t7"
real_stat_save_path = "./gt_stats/pointnet_max_pc_2048_emb_1024/%s-%s.npz" % (category, split)

model.load_state_dict(torch.load(path))
model.eval()

print('Loading all real data ...')
real_pts = np.load('./gt_stats/real_pcs/%s-%s-real-pcs.npy' % (category, split))
print('real_pts: ', real_pts.shape)

print('Calculating real statistics...')
b, _, _ = real_pts.shape
batch_size = 32
real_feature_list = []
bar = ProgressBar()
with torch.no_grad():
    for i in bar(range(b // batch_size + 1)):
        real_pts_batch = real_pts[i * batch_size : min((i + 1) * batch_size, b)]
        if real_pts_batch.shape[0] > 0:
            real_feature_batch = model(torch.Tensor(real_pts_batch).to(device))[1].cpu().detach().numpy()
            real_feature_list.append(real_feature_batch)
real_feature = np.concatenate(real_feature_list, 0)
real_mean = np.mean(real_feature, axis=0)
real_cov = np.cov(real_feature, rowvar=False)
np.savez(real_stat_save_path, mean=real_mean, cov=real_cov)
print("save real statistics to: ", real_stat_save_path)

