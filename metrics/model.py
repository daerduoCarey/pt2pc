import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self, output_channels=40, emb_dims=1024):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, emb_dims, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.mlp1 = nn.Linear(emb_dims, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, output_channels)
        
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dp = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.shape[0]

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.max(dim=-1)[0]
        feat = x

        x = torch.relu(self.bn6(self.mlp1(x)))
        x = torch.relu(self.bn7(self.mlp2(x)))
        x = self.dp(x)
        x = self.mlp3(x)
        return x, feat

