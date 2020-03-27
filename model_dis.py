import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'sampling'))
from data import Tree
from sampling import furthest_point_sample


class PointNet(nn.Module):
    def __init__(self, feat_len):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.mlp = nn.Linear(1024, feat_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        x = x.max(dim=-1)[0]
        
        x = F.leaky_relu(self.mlp(x))
        
        return x


class PartEncoder(nn.Module):

    def __init__(self, feat_len):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.fc = nn.Linear(feat_len, feat_len)
        
    def forward(self, pc):
        net = pc.transpose(2, 1)
        
        net = F.leaky_relu(self.conv1(net))
        net = F.leaky_relu(self.conv2(net))
        net = F.leaky_relu(self.conv3(net))
        net = F.leaky_relu(self.conv4(net))
        
        net = net.max(dim=2)[0]
        
        net = F.leaky_relu(self.fc(net))

        return net


class ChildrenEncoder(nn.Module):

    def __init__(self, feat_len, hidden_len, symmetric_type):
        super(ChildrenEncoder, self).__init__()

        print(f'Using Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type

        self.child_op = nn.Linear(feat_len + Tree.num_sem, hidden_len)
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


class Network(nn.Module):

    def __init__(self, args, device):
        super(Network, self).__init__()
        self.args = args
        self.device = device
        
        self.part_encoder = PartEncoder(args.feat_len)
        self.children_encoder = ChildrenEncoder(args.feat_len, args.hidden_len, args.symmetric_type)

        self.pointnet_encoder = PointNet(args.feat_len)
    
        self.mlp1 = nn.Linear(args.feat_len, 1)
        self.mlp2 = nn.Linear(args.feat_len, 1)

        self.final_activation = args.final_activation

    def encode(self, node, leaf_feats):
        if len(node.children) == 0:
            return leaf_feats[:, node.geo_id]
        else:
            child_feats = []
            batch_size = leaf_feats.shape[0]
            for cnode in node.children:
                cur_child_feat = torch.cat([self.encode(cnode, leaf_feats), cnode.get_semantic_one_hot().repeat(batch_size, 1)], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            return self.children_encoder(child_feats)

    def forward(self, node, pcs):
        batch_size = pcs.shape[0]
        part_cnt = pcs.shape[1]

        ### StructureNet Discriminator
        # encode all leaf geometry
        leaf_feats = self.part_encoder(pcs.view(batch_size * part_cnt, -1, 3)).view(batch_size, part_cnt, -1)

        # encode the hierarchy until root
        sn_net = self.encode(node, leaf_feats)

        ### holistic PointNet Discriminator
        # fps
        shape_pcs = pcs.view(batch_size, -1, 3)
        with torch.no_grad():
            shape_pc_id1 = torch.arange(batch_size).unsqueeze(1).repeat(1, self.args.num_point_per_shape).long().view(-1).to(self.device)
            shape_pc_id2 = furthest_point_sample(shape_pcs, self.args.num_point_per_shape).long().view(-1)
        shape_pcs = shape_pcs[shape_pc_id1, shape_pc_id2].view(batch_size, self.args.num_point_per_shape, 3)

        # pointnet encode
        pn_net = self.pointnet_encoder(shape_pcs)

        ### balance two discriminators
        # [SN Dis] produce real/fake scores
        sn_net = self.mlp1(sn_net).squeeze(1)
        if self.final_activation:
            sn_net = torch.sigmoid(sn_net)
        
        # [PN Dis] produce real/fake scores
        pn_net = self.mlp2(pn_net).squeeze(1)
        if self.final_activation:
            pn_net = torch.sigmoid(pn_net)

        # aggregate two scores
        net = sn_net + pn_net * self.args.pointnet_dis_score_multiplier
        
        return net, sn_net, pn_net

