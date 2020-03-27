import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from model import PointNet


class FID(object):
    
    def __init__(self, mode, dataset, device, split, path=None):
        if mode == "PointNet":
            self.model = PointNet().to(device)
            if path is None:
                path = "./metrics/pointnet_modelnet40/checkpoints/pointnet_max_pc_2048_emb_1024/models/model.t7"
            if split == 'train':
                self.real_stat_save_path = "./metrics/gt_stats/pointnet_max_pc_2048_emb_1024/%s-train.npz" % dataset
            elif split == 'test':
                self.real_stat_save_path = "./metrics/gt_stats/pointnet_max_pc_2048_emb_1024/%s-test.npz" % dataset
            else:
                raise ValueError('ERROR: unknown split %s!' % split)
            print('Using PointNet, gt_stat_fn: %s\n' % self.real_stat_save_path)

        else:
            raise ValueError('ERROR: unknown FID mode %s!' % mode)

        self.model.load_state_dict(torch.load(path))
        self.model = self.model.eval()
        self.device = device

    def get_fid(self, fake_pts, batch_size=32):
        f = np.load(self.real_stat_save_path)
        real_mean, real_cov = f['mean'], f['cov']

        fake_feature_list = []
        with torch.no_grad():
            b, _, _ = fake_pts.shape
            for i in range(b // batch_size + 1):
                fake_pts_batch = fake_pts[i * batch_size : min((i + 1) * batch_size, b)]
                if fake_pts_batch.shape[0] > 0:
                    fake_feature_batch = self.model(torch.Tensor(fake_pts_batch).to(self.device))[1].cpu().detach().numpy()
                    fake_feature_list.append(fake_feature_batch)
        fake_feature = np.concatenate(fake_feature_list, 0)
        fake_mean = np.mean(fake_feature, axis=0)
        fake_cov = np.cov(fake_feature, rowvar=False)
        
        fid = self.calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        return fid        

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
    
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
    
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'
    
        diff = mu1 - mu2
    
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
    
        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)       

