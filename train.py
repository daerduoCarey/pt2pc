import os
import sys
import shutil
import json
import time
import random
import datetime
import argparse
import torch
import numpy as np
import tensorboardX
import utils
from data import Tree, PartGraphShapesDataset
import trainer as trainer_def
import model_gen as gen_model_def
import model_dis as dis_model_def

### parameter setup
parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, help='data category')
parser.add_argument('--exp_suffix', type=str, help='name suffix of the training run', default='nothing')
#parser.add_argument('--seed', type=int, help='random seed (for reproducibility) [-1 means to randomly sample one]', default=3124256514)
parser.add_argument('--seed', type=int, help='random seed (for reproducibility) [-1 means to randomly sample one]', default=-1)
parser.add_argument("--resume_ckpt", type=str, help='if to resume, specify the ckpt file [default: None, meaning train from scratch]', default=None)

# parameters for gan training
parser.add_argument('--dataset_mode', type=str, help='[gan] dataset mode [sample_by_template, sample_by_shape]', default='sample_by_shape')
parser.add_argument("--X", type=int, help='[gan] number of templates in a batch', default=6)
parser.add_argument("--Y", type=int, help='[gan] number of shapes per template in a batch', default=3)
parser.add_argument("--lr", type=float, help='[gan] learning rate', default=0.0001)
parser.add_argument('--num_workers', type=int, help='[gan] number of worker threads for data loading', default=6)
parser.add_argument("--n_critic", type=int, help='[gan] number of dis training steps per gen training step', default=10)
parser.add_argument("--max_epochs", type=int, help='[gan] max number of epochs to train', default=1000000)
parser.add_argument("--epochs_per_eval", type=int, help='[gan] number of training epochs per evaluation', default=50)
parser.add_argument('--num_visu', type=int, help='[gan] number of generated outputs to visualize [default: None --> no visu output]', default=5)

# Metrics related parameters
parser.add_argument("--epochs_per_metric", type=int, help='[metric] number of training epochs per metric computation', default=10)
parser.add_argument("--num_fake_per_metric", type=int, help='[metric] number of fake examples to generate per metric evaluation', default=1000)
parser.add_argument("--fid_mode", type=str, help='[metric] FID score mode [PointNet, DGCNN]', default='PointNet')
parser.add_argument("--num_point_per_shape", type=int, help='[gen] number of points per shape for fpd', default=2048)

# parameters for generator
parser.add_argument("--z_dim", type=int, help='[gen] the gaussian noise z dimension', default=256)
parser.add_argument("--num_point_per_part", type=int, help='[gen] number of points per part', default=1000)
parser.add_argument("--max_part_per_parent", type=int, help='[gen] max part per parent', default=10)
parser.add_argument("--template_feat_len", type=int, help='[gen] template feature length', default=64)
parser.add_argument("--template_symmetric_type", type=str, help='[gen] template symmetric type', default='max')

# parameters for discriminator
parser.add_argument("--pointnet_dis_score_multiplier", type=float, help='[dis] pointnet_dis_score_multiplier', default=1.0)
parser.add_argument("--pooling_type", type=str, help='[dis] pooling type [max, avg, or mix]', default='max')
parser.add_argument("--symmetric_type", type=str, help='[dis] symmetric type', default='max')
parser.add_argument("--final_activation", action='store_true', help='[dis] final sigmoid activation', default=False)

# shared parameters
parser.add_argument("--feat_len", type=int, help='[gen/dis] feature length', default=256)
parser.add_argument("--hidden_len", type=int, help='[gen/dis] hidden length', default=256)

# loss weights
parser.add_argument('--loss_weight_gp', type=float, help='[gan] coefficient for gradient-penalty loss term', default=1.0)

# parse args
args = parser.parse_args()

# generate other parameters
args.exp_name = 'exp_%s_%s' % (args.category, args.exp_suffix)
args.data_dir = './data/%s_geo' % args.category
args.pg_dir = './stats/part_trees/%s_all_no_other_less_than_10_parts-train' % args.category

with open(os.path.join(args.pg_dir, 'visu_pg_list.txt'), 'r') as fin:
    args.visu_pg_list = [int(l.rstrip()) for l in fin.readlines()]

### preparation
# load category information
Tree.load_category_info(args.category)

# check if training run already exists. If so, delete it.
if os.path.exists(os.path.join('log', args.exp_name)):
    response = input('A training run named "%s" already exists, overwrite? (y/n) ' % args.exp_name)
    if response != 'y':
        sys.exit()
if os.path.exists(os.path.join('log', args.exp_name)):
    shutil.rmtree(os.path.join('log', args.exp_name))

# create directories for this run
if not os.path.exists(os.path.join('log', args.exp_name)):
    os.makedirs(os.path.join('log', args.exp_name))

# file log
flog = open(os.path.join('log', args.exp_name, 'train_log.txt'), 'w')

# backup command
flog.write(' '.join(sys.argv) + '\n')

# set training device
device = torch.device('cuda:0')

# control randomness
if args.seed < 0:
    args.seed = random.randint(1, 10000)
print('Random Seed: %d' % args.seed)
flog.write(f'Random Seed: {args.seed}\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# save config
torch.save(args, os.path.join('log', args.exp_name, 'conf.pth'))


### main procedure
# get models
generator = gen_model_def.Network(args, device).to(device)
utils.printout(flog, str(generator))
discriminator = dis_model_def.Network(args, device).to(device)
utils.printout(flog, str(discriminator))

# create dataset and dataloader
train_dataset = PartGraphShapesDataset(args.data_dir, args.pg_dir, device, args.Y, mode=args.dataset_mode)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.X, shuffle=True, \
        num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn, collate_fn=utils.collate_feats)

# tensorboard logger
logger = tensorboardX.SummaryWriter(log_dir=os.path.join('log', args.exp_name))

# get a gan trainer
trainer = trainer_def.Trainer(args.exp_name, generator, discriminator, \
        args=args, device=device, flog=flog, logger=logger)

# if to resume
if args.resume_ckpt is None:
    start_epoch = 0
else:
    start_epoch = int(args.resume_ckpt.split('/')[-1].split('.')[0].split('_')[1])
start_iteration = start_epoch * (len(train_dataset) // args.X)
print('start_epoch: %d, start_iteration: %d' % (start_epoch, start_iteration))
flog.write('start_epoch: %d, start_iteration: %d\n' % (start_epoch, start_iteration))

if start_epoch != 0:
    print("Loading checkpoints from: %s" % (args.resume_ckpt))
    flog.write("Loading checkpoints from: %s\n" % (args.resume_ckpt))
    trainer.load_model(args.resume_ckpt)

# train
trainer.train(train_dataset, train_dataloader, \
        start_iteration=start_iteration, start_epoch=start_epoch)

# exit
flog.close()

