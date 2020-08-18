"""
    Generate n shapes using the given input part-graph template
    The input template is given as either an part-graph index in the training data, or a customized part-graph json file
"""

import os
import sys
import shutil
import json
import argparse
import torch
import numpy as np
import utils
from subprocess import call
from data import Tree, PartGraphShapesDataset
import model_gen as gen_model_def


### parameter setup
parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, help='category')
parser.add_argument('--pg_idx', type=int, help='pg index to test [default: None]', default=None)
parser.add_argument('--pg_json', type=str, help='a template_with_group_id.json file [default: None]', default=None)
parser.add_argument('--num_gen', type=int, help='number of generated outputs', default=10)
parser.add_argument('--num_real', type=int, help='number of the first-k real shapes to show', default=10)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--y', action='store_true', help='overwrite if out_dir exists [default: False]', default=False)

# parameters for generator
parser.add_argument("--z_dim", type=int, help='[gen] the gaussian noise z dimension', default=256)
parser.add_argument("--num_point_per_part", type=int, help='[gen] number of points per part', default=1000)
parser.add_argument("--max_part_per_parent", type=int, help='[gen] max part per parent', default=10)
parser.add_argument("--template_feat_len", type=int, help='[gen] template feature length', default=64)
parser.add_argument("--template_symmetric_type", type=str, help='[gen] template symmetric type', default='max')

# shared parameters
parser.add_argument("--feat_len", type=int, help='[gen/dis] feature length', default=256)
parser.add_argument("--hidden_len", type=int, help='[gen/dis] hidden length', default=256)

# parse parameters
args = parser.parse_args()

# generate other parameters
args.data_dir = './data/%s_geo' % args.category
args.pg_dir = './stats/part_trees/%s_all_no_other_less_than_10_parts-train' % args.category

### preparation
# load category information
Tree.load_category_info(args.category)

# output directory
if args.pg_idx is not None:
    out_dir = os.path.join('pretrained_ckpts', 'test-%s-%04d' % (args.category, args.pg_idx))
elif args.pg_json is not None:
    out_dir = os.path.join('pretrained_ckpts', 'test-%s-%s' % (args.category, args.pg_json.split('.')[0]))
else:
    print('ERROR: you have to specify pg_idx or pg_json!')
    exit(1)

if os.path.exists(out_dir):
    if not args.y:
        response = input('Output directory "%s" already exists, overwrite? (y/n) ' % out_dir)
        if response != 'y':
            exit(0)
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

# set training device
device = torch.device('cuda:0')


### main procedure
# get models
generator = gen_model_def.Network(args, device).to(device)

# load pretrained model
ckpt_fn = os.path.join('pretrained_ckpts', args.category+'.ckpt')
print('Loading ckpt from %s ...' % ckpt_fn)
generator.load_state_dict(torch.load(ckpt_fn)['generator_state_dict'])

# get part-graph template
if args.pg_idx is not None:
    dataset = PartGraphShapesDataset(args.data_dir, args.pg_dir, device, 1)
    pg_template = dataset.get_pg_template(args.pg_idx)
    pg_template_fn = os.path.join(args.pg_dir, 'pt-%d' % args.pg_idx, 'template.json')

    # output some real shapes
    for i in range(args.num_real):
        fn = 'real-%04d' % i
        name, real_pcs = dataset.get_pg_real_pc(args.pg_idx, i)
        with open(os.path.join(out_dir, fn + '.txt'), 'w') as fout:
            fout.write('%s\n' % name)
        real_pcs = real_pcs.cpu().detach().numpy()
        utils.export_part_pcs(os.path.join(out_dir, fn), real_pcs)

else:
    pg_template_fn = args.pg_json
    pg_template = Tree.load_template(pg_template_fn, device)
    leaf_ids = pg_template.get_leaf_ids()
    pg_template.mark_geo_id({y: x for x, y in enumerate(leaf_ids)})
    pg_template.leaf_cnt = len(leaf_ids)

# save pg_template_fn
cmd = 'cp %s %s/template.json' % (pg_template_fn, out_dir)
call(cmd, shell=True)

# forward through the network to generate shapes
num_batch = args.num_gen // args.batch_size
for i in range(num_batch+1):
    if i == num_batch:
        cur_num_gen = args.num_gen % args.batch_size
    else:
        cur_num_gen = args.batch_size
    
    zs = torch.randn(cur_num_gen, args.z_dim).to(device)
    pcs = generator(pg_template, zs).cpu().detach().numpy()
    
    for j in range(cur_num_gen):
        fn = 'generated-%04d' % (i * args.batch_size + j)
        utils.export_part_pcs(os.path.join(out_dir, fn), pcs[j])

