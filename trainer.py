import os
import sys
import torch
import time
import math
import numpy as np
from torch.autograd import Variable
from utils import render_part_pcs, export_part_pcs, render_pc, export_pc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'metrics'))
sys.path.append(os.path.join(BASE_DIR, 'sampling'))
from fid import FID
from subprocess import call
from sampling import furthest_point_sample


class Trainer(object):
    def __init__(self, exp_name, generator, discriminator, args, device, flog, logger):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.exp_name = exp_name
        self.args = args
        self.device = device
        self.flog = flog
        self.logger = logger
        self.fid = FID(self.args.fid_mode, self.args.category, device, 'train')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

    def compute_gradient_penalty(self, pg_node, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interp_samples = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        interp_score, _, _ = self.discriminator.forward(pg_node, interp_samples)
        fake = torch.ones(interp_score.size()).to(self.device)
        
        gradients = torch.autograd.grad(
            outputs=interp_score,
            inputs=interp_samples,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.contiguous().view(batch_size, -1)
        gradient_penalty = ((gradients.pow(2).sum(dim=1) + 1e-4).sqrt() - 1) ** 2
        return gradient_penalty

    def train_iteration(self, dataset, data, iteration):
        num_pg = len(data[0])
        num_shape_per_pg = data[1][0].shape[0]
        
        # get all pg-templates
        pg_templates = []
        for i in range(num_pg):
            pg_templates.append(dataset.get_pg_template(data[0][i]))

        # train discriminator
        self.discriminator.train()
        self.generator.eval()

        self.optimizer_d.zero_grad()

        real_score = []; fake_score = []; gradient_penalty = [];
        real_sn_score = []; real_pn_score = [];
        fake_sn_score = []; fake_pn_score = [];
        for i in range(num_pg):
            with torch.no_grad():
                zs = torch.randn(num_shape_per_pg, self.args.z_dim).to(self.device)
                fake_part_pcs = self.generator(pg_templates[i], zs).detach()
            real_part_pcs = Variable(torch.Tensor(data[1][i]).to(self.device))
        
            cur_real_score, cur_real_sn_score, cur_real_pn_score = self.discriminator(pg_templates[i], real_part_pcs)
            real_score.append(cur_real_score); real_sn_score.append(cur_real_sn_score); real_pn_score.append(cur_real_pn_score);
            cur_fake_score, cur_fake_sn_score, cur_fake_pn_score = self.discriminator(pg_templates[i], fake_part_pcs)
            fake_score.append(cur_fake_score); fake_sn_score.append(cur_fake_sn_score); fake_pn_score.append(cur_fake_pn_score);
            
            gradient_penalty.append(self.compute_gradient_penalty(pg_templates[i], real_part_pcs.data, fake_part_pcs.data))

        real_score = torch.cat(real_score); real_sn_score = torch.cat(real_sn_score); real_pn_score = torch.cat(real_pn_score);
        self.logger.add_scalar('real_score', torch.mean(real_score).item(), iteration)
         
        fake_score = torch.cat(fake_score); fake_sn_score = torch.cat(fake_sn_score); fake_pn_score = torch.cat(fake_pn_score);
        self.logger.add_scalar('fake_score', torch.mean(fake_score).item(), iteration)
        
        gradient_penalty = torch.cat(gradient_penalty)
        gradient_penalty = torch.mean(gradient_penalty)
        self.logger.add_scalar("gradient_penalty", gradient_penalty.item(), iteration)
        
        wasserstein_estimate = torch.mean(real_score) - torch.mean(fake_score)
        self.logger.add_scalar('wasserstein_estimate', wasserstein_estimate.item(), iteration)
        wasserstein_estimate_sn = torch.mean(real_sn_score) - torch.mean(fake_sn_score)
        self.logger.add_scalar('wasserstein_estimate_sn', wasserstein_estimate_sn.item(), iteration)
        wasserstein_estimate_pn = torch.mean(real_pn_score) - torch.mean(fake_pn_score)
        self.logger.add_scalar('wasserstein_estimate_pn', wasserstein_estimate_pn.item(), iteration)
        
        d_loss = self.args.loss_weight_gp * gradient_penalty - wasserstein_estimate
        self.logger.add_scalar('train_d_loss', d_loss.item(), iteration)

        d_loss.backward()
        self.optimizer_d.step()
            
        out_str = '  **Training DIS %s** [w_dist: %.4f] [real_scores: %.4f] [fake_scores: %.4f] [gp: %.4f]' \
                % (self.exp_name, wasserstein_estimate.item(), torch.mean(real_score).item(), torch.mean(fake_score).item(), gradient_penalty.item())
        print(out_str)
        self.flog.write(out_str + '\n')
        
        if iteration % self.args.n_critic == 0:
            # train generator
            self.discriminator.eval()
            self.generator.train()

            self.optimizer_g.zero_grad()

            fake_score = [];
            for i in range(num_pg):
                zs = torch.randn(num_shape_per_pg, self.args.z_dim).to(self.device)
                fake_part_pcs = self.generator(pg_templates[i], zs)
               
                cur_fake_score, _, _ = self.discriminator(pg_templates[i], fake_part_pcs)
                fake_score.append(cur_fake_score)

            fake_score = torch.cat(fake_score)
            
            g_loss = - torch.mean(fake_score)
            self.logger.add_scalar('train_g_loss', g_loss.item(), iteration)
            
            g_loss.backward()
            self.optimizer_g.step()
            
            out_str = '  **Training GEN %s** [fake_scores: %.4f]' \
                    % (self.exp_name, torch.mean(fake_score).item())
            print(out_str)
            self.flog.write(out_str + '\n')
     
    def eval_metric(self, dataset, epoch):
        self.generator.eval()
        
        # generate fake pcs
        with torch.no_grad():
            fake_pcs = []
            for i in range(self.args.num_fake_per_metric):
                idx = np.random.choice(len(dataset))
                pg_idx, _ = dataset[idx]
                pg_template = dataset.get_pg_template(pg_idx)
                z = torch.randn(1, self.args.z_dim).to(self.device)
                gen_part_pc = self.generator(pg_template, z)
                gen_pc = gen_part_pc.reshape(1, -1, 3)
                gen_pc_idx = furthest_point_sample(gen_pc, self.args.num_point_per_shape)[0]
                gen_pc = gen_pc[0, gen_pc_idx.long()]
                gen_pc = gen_pc.cpu().detach().numpy()
                fake_pcs.append(np.expand_dims(gen_pc, 0))
            fake_pcs = np.concatenate(fake_pcs, 0)

        # compute FPD score
        fpd = self.fid.get_fid(fake_pcs)
        self.logger.add_scalar('eval_fpd', fpd, epoch)
        out_str = '##Eval Metric %s## [fpd: %.4f]' % (self.exp_name, fpd)
        print(out_str)
        self.flog.write(out_str + '\n')

    def train(self, train_dataset, train_dataloader, start_iteration=0, start_epoch=0):
        iteration = start_iteration
        for epoch in range(start_epoch, self.args.max_epochs):
            # train one epoch
            out_str = '\n %s [Epoch %03d/%03d]' % (time.asctime(time.localtime(time.time())), epoch, self.args.max_epochs)
            print(out_str)
            self.flog.write(out_str + '\n')
            
            for i, data in enumerate(train_dataloader):
                self.train_iteration(train_dataset, data, iteration)
                iteration = iteration + 1

            if (epoch + 1) % self.args.epochs_per_metric == 0:
                self.eval_metric(train_dataset, epoch)

            if (epoch + 1) % self.args.epochs_per_eval == 0:
                self.discriminator.eval()
                self.generator.eval()

                with torch.no_grad():
                    # save checkpoint
                    out_fn = os.path.join('log', self.args.exp_name, 'model_%06d.ckpt' % epoch)
                    out_str = 'Saving checkpoint to %s' % out_fn
                    print(out_str)
                    self.flog.write(out_str + '\n')
                    torch.save({
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'discriminator_optimizer_state_dict': self.optimizer_d.state_dict(),
                        'generator_state_dict': self.generator.state_dict(),
                        'generator_optimizer_state_dict': self.optimizer_g.state_dict(),
                    }, out_fn)

                    # visualize current results
                    if self.args.num_visu is not None:
                        cur_visu_dir = os.path.join('log', self.args.exp_name, 'visu-%08d' % epoch)
                        os.mkdir(cur_visu_dir)
                        cur_gen_dir = os.path.join(cur_visu_dir, 'gen')
                        os.mkdir(cur_gen_dir)
                        cur_gen2_dir = os.path.join(cur_visu_dir, 'gen2')
                        os.mkdir(cur_gen2_dir)
                        cur_real_dir = os.path.join(cur_visu_dir, 'real')
                        os.mkdir(cur_real_dir)
                        cur_info_dir = os.path.join(cur_visu_dir, 'info')
                        os.mkdir(cur_info_dir)
                        print('Visualizing ...')
                        self.flog.write('Visualizing ...\n')
                        for pg_idx in self.args.visu_pg_list:
                            pg_node = train_dataset.get_pg_template(pg_idx)
                            zs = torch.randn(self.args.num_visu, self.args.z_dim).to(self.device)
                            part_pcs = self.generator(pg_node, zs)
                            shape_pcs = part_pcs.view(self.args.num_visu, -1, 3)
                            shape_pc_id1 = torch.arange(self.args.num_visu).unsqueeze(1).repeat(1, self.args.num_point_per_shape).long().view(-1).to(self.device)
                            shape_pc_id2 = furthest_point_sample(shape_pcs, self.args.num_point_per_shape).long().view(-1)
                            shape_pcs = shape_pcs[shape_pc_id1, shape_pc_id2].view(self.args.num_visu, self.args.num_point_per_shape, 3)
                            real_names, real_part_pcs = train_dataset.get_pg_real_pcs(pg_idx, self.args.num_visu)
                            part_pcs = part_pcs.cpu().detach().numpy()
                            real_part_pcs = real_part_pcs.cpu().detach().numpy()
                            for pcid in range(self.args.num_visu):
                                fn = 'pg-%04d-shape-%04d' % (pg_idx, pcid)
                                render_part_pcs([part_pcs[pcid]], title_list=['shape-%04d' % pcid],
                                        out_fn=os.path.join(cur_gen_dir, fn+'.png'))
                                export_part_pcs(os.path.join(cur_gen_dir, fn), part_pcs[pcid])
                                render_part_pcs([real_part_pcs[pcid]], title_list=['shape-%04d' % pcid],
                                        out_fn=os.path.join(cur_real_dir, fn+'.png'))
                                export_part_pcs(os.path.join(cur_real_dir, fn), real_part_pcs[pcid])
                                cur_shape_pc = shape_pcs[pcid].cpu().detach().numpy()
                                render_pc(os.path.join(cur_gen2_dir, fn+'.png'), cur_shape_pc)
                                export_pc(os.path.join(cur_gen2_dir, fn+'.obj'), cur_shape_pc)
                                with open(os.path.join(cur_info_dir, fn+'.txt'), 'w') as fout:
                                    fout.write('%s\n' % real_names[pcid])
                        sublist = 'gen,gen2,real,info'
                        cmd = 'cd %s && python %s . %d htmls %s %s > /dev/null' % (cur_visu_dir, \
                                os.path.join(BASE_DIR, 'gen_html_hierachy_local.py'), self.args.num_visu, sublist, sublist)
                        call(cmd, shell=True)
            
            self.flog.flush()

