import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from data import Tree
import utils


class TemplateChildrenEncoder(nn.Module):

    def __init__(self, feat_len, hidden_len, symmetric_type, max_part_per_parent):
        super(TemplateChildrenEncoder, self).__init__()

        print(f'Using Template Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type

        self.child_op = nn.Linear(feat_len + Tree.num_sem + max_part_per_parent, hidden_len)
        self.second = nn.Linear(hidden_len, feat_len)

    def forward(self, child_feats):
        child_feats = F.leaky_relu(self.child_op(child_feats))

        if self.symmetric_type == 'max':
            parent_feat = child_feats.max(dim=1)[0]
        elif self.symmetric_type == 'sum':
            parent_feat = child_feats.sum(dim=1)
        elif self.symmetric_type == 'avg':
            parent_feat = child_feats.mean(dim=1)
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        parent_feat = F.leaky_relu(self.second(parent_feat))

        return parent_feat


class TemplateEncoder(nn.Module):

    def __init__(self, feat_len, hidden_len, symmetric_type, max_part_per_parent, device):
        super(TemplateEncoder, self).__init__()
        self.feat_len = feat_len
        self.device = device
        self.max_part_per_parent = max_part_per_parent

        self.template_children_encoder = TemplateChildrenEncoder(feat_len, hidden_len, symmetric_type, max_part_per_parent)

    def encode(self, node):
        if len(node.children) == 0:
            ret = torch.zeros(1, self.feat_len).to(self.device)
        else:
            child_feats = []
            for cnode in node.children:
                cur_child_feat = torch.cat([self.encode(cnode), \
                        cnode.get_semantic_one_hot(), \
                        cnode.get_group_ins_one_hot(self.max_part_per_parent)], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))

            child_feats = torch.cat(child_feats, dim=1)

            ret = self.template_children_encoder(child_feats)
        
        node.template_feature = ret
        return ret

    def forward(self, node):
        return self.encode(node)


class PartDecoder(nn.Module):

    def __init__(self, feat_len):
        super(PartDecoder, self).__init__()

        self.mlp1 = nn.Linear(feat_len + 3, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 3)

    def forward(self, feat, pc):
        num_point = pc.shape[0]
        batch_size = feat.shape[0]

        net = torch.cat([feat.unsqueeze(dim=1).repeat(1, num_point, 1), \
                pc.unsqueeze(dim=0).repeat(batch_size, 1, 1)], dim=-1)
        
        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)

        return net


class Network(nn.Module):
    def __init__(self, args, device):
        super(Network, self).__init__()

        self.template_encoder = TemplateEncoder(args.template_feat_len, args.hidden_len, args.template_symmetric_type, args.max_part_per_parent, device)
        self.part_decoder = PartDecoder(args.feat_len)

        self.max_part_per_parent = args.max_part_per_parent

        self.mlp1 = nn.Linear(args.feat_len + Tree.num_sem + args.template_feat_len + self.max_part_per_parent, args.hidden_len)
        self.mlp2 = nn.Linear(args.hidden_len, args.feat_len)

        self.register_buffer('base_pc', torch.from_numpy(utils.load_pts('cube.pts')))

    def decode(self, node, feat, leaf_feats):
        if len(node.children) == 0:
            leaf_feats[node.geo_id] = feat.unsqueeze(dim=1)
        else:
            batch_size = feat.shape[0]

            for cnode in node.children:
                net = torch.cat([feat, \
                        cnode.get_semantic_one_hot().repeat(batch_size, 1), \
                        cnode.template_feature.repeat(batch_size, 1), \
                        cnode.get_group_ins_one_hot(self.max_part_per_parent).repeat(batch_size, 1)], dim=1)
                
                net = F.leaky_relu(self.mlp1(net))
                net = F.leaky_relu(self.mlp2(net))

                self.decode(cnode, net, leaf_feats)

    def forward(self, node, z):
        batch_size = z.shape[0]

        # compute part-tree subtree features
        self.template_encoder(node)

        # condition z on the root template feature
        feat = torch.cat([z, \
                    node.get_semantic_one_hot().repeat(batch_size, 1), \
                    node.template_feature.repeat(batch_size, 1), \
                    node.get_group_ins_one_hot(self.max_part_per_parent).repeat(batch_size, 1)], dim=1)
            
        feat = F.leaky_relu(self.mlp1(feat))
        feat = F.leaky_relu(self.mlp2(feat))

        # recursively decode node features until leaf parts
        leaf_feats = [None] * node.leaf_cnt
        self.decode(node, feat, leaf_feats)
        leaf_feats = torch.cat(leaf_feats, dim=1)

        # decode leaf feats to leaf pcs
        pcs = self.part_decoder(leaf_feats.view(batch_size * node.leaf_cnt, -1), self.base_pc).view(batch_size, node.leaf_cnt, -1, 3)

        return pcs

