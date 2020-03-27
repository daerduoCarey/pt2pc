"""
    This file defines minimal Tree/Node class for the PartGraph Shapes dataset
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils import data
from collections import namedtuple
import utils


# store a part hierarchy of graphs for a shape
class Tree(object):

    # global object category information
    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = None
    root_sem = None

    @ staticmethod
    def load_category_info(cat):
        with open(os.path.join('./stats/part_semantics/', cat+'.txt'), 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if '/' in y:
                    Tree.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        Tree.num_sem = len(Tree.part_name2id) + 1
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)
        Tree.root_sem = Tree.part_id2name[1]


    # store a part node in the tree
    class Node(object):

        def __init__(self, device, part_id=None, label=None, full_label=None, group_id=None, group_ins_id=None):
            self.device = device            # device that this node lives
            self.part_id = part_id          # part_id in result_after_merging.json of PartNet
            self.group_id = group_id        # group_id is 0, 1, 2, ...; it will be the same for equivalent subtree nodes
            self.group_ins_id = group_ins_id# group_ins_id is 0, 1, 2, ... within each equivalent class
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            self.children = []              # initialize to be empty (no children)
            self.geo_id = None              # the index of the part pc geo array
        
        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]
            
        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            out[0, Tree.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.device)
            
        def get_group_ins_one_hot(self, max_part_per_parent):
            out = np.zeros((1, max_part_per_parent), dtype=np.float32)
            out[0, self.group_ins_id] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.device)
            
        def _to_str(self, level, pid):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + \
                    (' [LEAF %d] ' % self.geo_id if len(self.children) == 0 else '    ') + \
                    '{part_id: %d, group_id: %d [%d], subtree_geo_ids: %s}\n' % \
                    (self.part_id, self.group_id, self.group_ins_id, str(self.subtree_geo_ids)) 
            for idx, child in enumerate(self.children):
                out_str += child._to_str(level+1, idx)
            return out_str

        def __str__(self):
            return self._to_str(0, 0)

        def get_leaf_ids(self):
            leaf_ids = []
            if len(self.children) == 0:
                leaf_ids.append(self.part_id)
            else:
                for cnode in self.children:
                    leaf_ids += cnode.get_leaf_ids()
            return leaf_ids

        def mark_geo_id(self, d):
            if self.part_id in d:
                self.geo_id = d[self.part_id]
            for cnode in self.children:
                cnode.mark_geo_id(d)

        def compute_subtree_geo_ids(self):
            if len(self.children) == 0:
                self.subtree_geo_ids = [self.geo_id]
            else:
                self.subtree_geo_ids = []
                for cnode in self.children:
                    self.subtree_geo_ids += cnode.compute_subtree_geo_ids()
            return self.subtree_geo_ids
    
    @staticmethod
    def load_template(fn, device):
        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(device=device,
                part_id=node_json['id'],
                group_id=node_json['group_id'],
                group_ins_id=node_json['group_ins_id'],
                label=node_json['label'])

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        return root


# extend torch.data.Dataset class for PartNet
class PartGraphShapesDataset(data.Dataset):

    def __init__(self, data_dir, pg_dir, device, batch_size, mode='sample_by_template'):
        self.data_dir = data_dir
        self.pg_dir = pg_dir
        self.device = device
        self.batch_size = batch_size
        self.mode = mode

        self.pg_shapes = []
        self.sample_by_shape_pgids = []
        with open(os.path.join(pg_dir, 'info.txt'), 'r') as fin:
            for i, l in enumerate(fin.readlines()):
                cur_pg_shapes = l.rstrip().split()
                self.pg_shapes.append(cur_pg_shapes)
                self.sample_by_shape_pgids += [i] * len(cur_pg_shapes)

        self.pg_templates = []
        self.pg_leaf_ids = []
        self.leaf_mappings = []
        for i in range(len(self.pg_shapes)):
            cur_pg_dir = os.path.join(pg_dir, 'pt-%d' % i)
            t = Tree.load_template(os.path.join(cur_pg_dir, 'template.json'), device)
            self.pg_templates.append(t)
            leaf_ids = t.get_leaf_ids()
            t.leaf_cnt = len(leaf_ids)
            self.pg_leaf_ids.append(leaf_ids)
            t.mark_geo_id({y: x for x, y in enumerate(self.pg_leaf_ids[i])})
            t.compute_subtree_geo_ids()

            self.leaf_mappings.append([])
            for anno_id in self.pg_shapes[i]:
                with open(os.path.join(cur_pg_dir, anno_id+'.txt'), 'r') as fin:
                    tmp_dict = dict()
                    for l in fin.readlines():
                        x, y = l.rstrip().split()
                        tmp_dict[int(x)] = int(y)
                cur_leaf_mapping = [tmp_dict[x] for x in self.pg_leaf_ids[i]]
                cur_leaf_mapping = np.array(cur_leaf_mapping, dtype=np.int32)
                self.leaf_mappings[i].append(cur_leaf_mapping)

            self.pg_leaf_ids[i] = np.array(self.pg_leaf_ids[i], dtype=np.int32)
        
        print('[PartGraphShapesDataset %d %s %d %d] %s %s' % (batch_size, mode, \
                len(self.pg_shapes), len(self.sample_by_shape_pgids), data_dir, pg_dir))

    def __len__(self):
        if self.mode == 'sample_by_template':
            return len(self.pg_shapes)
        elif self.mode == 'sample_by_shape':
            return len(self.sample_by_shape_pgids)
        else:
            raise ValueError('ERROR: unknown mode %s!' % self.mode)

    def get_pg_shapes(self, index):
        return self.pg_shapes[index]

    def get_pg_template(self, index):
        return self.pg_templates[index]

    def get_pg_leaf_ids(self, index):
        return self.pg_leaf_ids[index]

    def get_pg_real_pcs(self, index, num_shape):
        ids = np.random.choice(len(self.pg_shapes[index]), num_shape, replace=True)
        part_pcs = np.zeros((num_shape, len(self.pg_leaf_ids[index]), 1000, 3), dtype=np.float32)
        names = []
        for i, idx in enumerate(ids):
            geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][idx] + '.npz')
            geo_data = np.load(geo_fn)['parts']
            part_pcs[i] = geo_data[self.leaf_mappings[index][idx]]
            names.append(self.pg_shapes[index][idx])
        out = torch.from_numpy(part_pcs)
        return (names, out)
    
    def get_pg_real_pc(self, index, j):
        j = j % len(self.pg_shapes[index])
        geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][j] + '.npz')
        geo_data = np.load(geo_fn)['parts']
        part_pcs = geo_data[self.leaf_mappings[index][j]]
        out = torch.from_numpy(part_pcs)
        return self.pg_shapes[index][j], out

    def __getitem__(self, index):
        if self.mode == 'sample_by_shape':
            index = self.sample_by_shape_pgids[index]
        ids = np.random.choice(len(self.pg_shapes[index]), self.batch_size, replace=True)
        part_pcs = np.zeros((self.batch_size, len(self.pg_leaf_ids[index]), 1000, 3), dtype=np.float32)
        for i, idx in enumerate(ids):
            geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][idx] + '.npz')
            geo_data = np.load(geo_fn)['parts']
            part_pcs[i] = geo_data[self.leaf_mappings[index][idx]]
        out = torch.from_numpy(part_pcs)
        return (index, out)


# PartNet Entire-shape Point-cloud Dataset (for training holistic-pc-gan baselines)
class PartNetShapeDataset(data.Dataset):

    def __init__(self, data_dir, object_list, num_point=2048):
        print('[PartNetShapeDataset %d] %s %s' % (num_point, data_dir, object_list))
        self.data_dir = data_dir
        self.num_point = num_point

        with open(object_list, 'r') as fin:
            self.shape_ids = [int(l.rstrip()) for l in fin.readlines()]
        print('Total Data: ', len(self))

    def __len__(self):
        return len(self.shape_ids)

    def get_random_batch(self, batch_size):
        ids = np.random.choice(len(self.shape_ids), batch_size, replace=True)
        out = np.zeros((batch_size, self.num_point, 3), dtype=np.float32)
        names = []
        for i, idx in enumerate(ids):
            names.append(self.shape_ids[idx])
            pts = utils.load_pts(os.path.join(self.data_dir, str(self.shape_ids[idx]), 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))
            out[i] = pts[:self.num_point]
        out = torch.from_numpy(out).float()
        return names, out
    
    def __getitem__(self, index):
        pts = utils.load_pts(os.path.join(self.data_dir, str(self.shape_ids[index]), 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))
        out = torch.from_numpy(pts[:self.num_point]).float().unsqueeze(0)
        return (self.shape_ids[index], out)


# used for vanilla c-GAN experiments
class PartGraphWholeShapesDataset(data.Dataset):

    def __init__(self, data_dir, pg_dir, device, batch_size, num_point, mode='sample_by_template'):
        print('[PartGraphWholeShapesDataset %d %s %d] %s %s' % (batch_size, mode, num_point, data_dir, pg_dir))
        self.data_dir = data_dir
        self.pg_dir = pg_dir
        self.device = device
        self.batch_size = batch_size
        self.mode = mode
        self.num_point = num_point

        self.pg_shapes = []
        self.sample_by_shape_pgids = []
        with open(os.path.join(pg_dir, 'info.txt'), 'r') as fin:
            for i, l in enumerate(fin.readlines()):
                cur_pg_shapes = l.rstrip().split()
                self.pg_shapes.append(cur_pg_shapes)
                self.sample_by_shape_pgids += [i] * len(cur_pg_shapes)

        self.pg_templates = []
        for i in range(len(self.pg_shapes)):
            cur_pg_dir = os.path.join(pg_dir, 'pt-%d' % i)
            t = Tree.load_template(os.path.join(cur_pg_dir, 'template.json'), device)
            self.pg_templates.append(t)
            leaf_ids = t.get_leaf_ids()
            t.leaf_cnt = len(leaf_ids)
            t.mark_geo_id({y: x for x, y in enumerate(leaf_ids)})
    
    def __len__(self):
        if self.mode == 'sample_by_template':
            return len(self.pg_shapes)
        elif self.mode == 'sample_by_shape':
            return len(self.sample_by_shape_pgids)
        else:
            raise ValueError('ERROR: unknown mode %s!' % self.mode)

    def get_pg_shapes(self, index):
        return self.pg_shapes[index]

    def get_pg_template(self, index):
        return self.pg_templates[index]

    def get_pg_real_pcs(self, index, num_shape):
        ids = np.random.choice(len(self.pg_shapes[index]), num_shape, replace=True)
        names = []; out = np.zeros((num_shape, self.num_point, 3), dtype=np.float32);
        for i, idx in enumerate(ids):
            out[i] = utils.load_pts(os.path.join(self.data_dir, self.pg_shapes[index][idx], 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))[:self.num_point]
            names.append(self.pg_shapes[index][idx])
        out = torch.from_numpy(out)
        return (names, out)
 
    def get_pg_real_pc(self, index, j):
        out = utils.load_pts(os.path.join(self.data_dir, self.pg_shapes[index][j], 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))[:self.num_point]
        out = torch.from_numpy(out)
        return out
    
    def __getitem__(self, index):
        if self.mode == 'sample_by_shape':
            index = self.sample_by_shape_pgids[index]
        ids = np.random.choice(len(self.pg_shapes[index]), self.batch_size, replace=True)
        out = np.zeros((self.batch_size, self.num_point, 3), dtype=np.float32)
        for i, idx in enumerate(ids):
            out[i] = utils.load_pts(os.path.join(self.data_dir, self.pg_shapes[index][idx], 'point_sample', 'sample-points-all-pts-nor-rgba-10000.txt'))[:self.num_point]
        out = torch.from_numpy(out)
        return (index, out)

