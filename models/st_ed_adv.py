"""Copyright 2020 ETH Zurich, Yufeng Zheng, Seonwook Park
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from models.encoder import Encoder
from models.decoder import Decoder
from models.redirtrans import ReDirTrans_P, ReDirTrans_DP, Fusion_Layer
from encoder4editing_tmp.models.latent_codes_pool import LatentCodesPool
from encoder4editing_tmp.models.discriminator import LatentCodesDiscriminator
import losses
from models.gazeheadnet import GazeHeadNet
from models.gazeheadResnet import GazeHeadResNet
from models.load_pretrained_model import load_pretrained_model
from core import DefaultConfig
import numpy as np
import torch.nn.functional as F
import lpips

config = DefaultConfig()

class STED(nn.Module):

    def __init__(self):
        super(STED, self).__init__()
        self.configuration = []
        self.configuration += ([(2, config.size_2d_unit)] * config.num_2d_units) # [(2, 16)] do2: 16

        self.num_all_pseudo_labels = np.sum([dof for dof, _ in self.configuration]) # 2
        self.num_all_embedding_features = np.sum([(dof + 1) * num_feats for dof, num_feats in self.configuration]) # 3 * 16 = 48

        self.e4e_net, self.pretrained_arcface, self.e4e_transforms, self.r50_transform = load_pretrained_model()

        self.redirtrans_p = ReDirTrans_P(configuration=self.configuration).to("cuda")
        self.redirtrans_dp = ReDirTrans_DP(configuration=self.configuration).to("cuda")
        self.fusion = Fusion_Layer().to("cuda")

        self.encoder = Encoder(self.num_all_pseudo_labels, self.num_all_embedding_features, self.configuration, self.e4e_net.encoder).to("cuda")
        self.decoder = Decoder(self.num_all_embedding_features, self.e4e_net.decoder).to("cuda")

        self.lpips = lpips.LPIPS(net='alex').to("cuda")
        self.GazeHeadNet_train = GazeHeadNet().to("cuda") # vGG16 estimator. 
        self.GazeHeadNet_eval = GazeHeadResNet().to("cuda") # ResNet estimator.
        self.discriminator = LatentCodesDiscriminator(512, 4).to("cuda")
        self.real_w_pool = LatentCodesPool(config.w_pool_size)
        self.fake_w_pool = LatentCodesPool(config.w_pool_size)

        self.pretrained_arcface_params = []
        self.redirtrans_params = []
        self.fusion_params = []
        self.generator_params = []
        self.lpips_params = []
        self.GazeHeadNet_train_params = []
        self.GazeHeadNet_eval_params = []
        self.discriminator_params = []

        for name, param in self.named_parameters():
            # Do not modify the feature extractor.
            if 'pretrained_arcface' in name:
                self.pretrained_arcface_params.append(param)
            elif 'redirtrans' in name:
                self.redirtrans_params.append(param)
            elif 'fusion' in name:
                self.fusion_params.append(param)
            elif 'encoder' in name or 'decoder' in name or 'classifier' in name:
                self.generator_params.append(param)
            elif 'lpips' in name:
                self.lpips_params.append(param)
            elif 'GazeHeadNet_train' in name:
                self.GazeHeadNet_train_params.append(param)
            elif 'GazeHeadNet_eval' in name:
                self.GazeHeadNet_eval_params.append(param)
            elif 'discriminator' in name:
                self.discriminator_params.append(param)

        for param in self.pretrained_arcface_params:
            param.requires_grad = False
        for param in self.generator_params:
            param.requires_grad = False
        for param in self.lpips_params:
            param.requires_grad = False
        for param in self.GazeHeadNet_train_params:
            param.requires_grad = False
        for param in self.GazeHeadNet_eval_params:
            param.requires_grad = False
        for param in self.discriminator_params:
            param.requires_grad = False
        
        self.redirtrans_optimizer = optim.Adam(self.redirtrans_params, lr=config.lr, weight_decay=config.l2_reg)
        self.fusion_optimizer = optim.Adam(self.fusion_params, lr=config.lr, weight_decay=config.l2_reg)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator_params, lr=config.w_discriminator_lr)
        self.optimizers = [self.fusion_optimizer, self.redirtrans_optimizer, self.discriminator_optimizer]

        # Wrap optimizer instances with AMP (Compute faster)
        if config.use_apex:
            from apex import amp
            models = [self.pretrained_arcface, self.redirtrans_p, self.redirtrans_dp, self.fusion, self.encoder, self.decoder, self.lpips, self.GazeHeadNet_train, self.GazeHeadNet_eval, self.discriminator]
            models, self.optimizers = amp.initialize(models, self.optimizers, opt_level='O1', num_losses=len(self.optimizers))
            [self.redirtrans_optimizer, self.fusion_optimizer, self.discriminator] = self.optimizers
            [self.pretrained_arcface, self.redirtrans_p, self.redirtrans_dp, self.fusion, self.encoder, self.decoder, self.lpips, self.GazeHeadNet_train, self.GazeHeadNet_eval, self.discriminator] = models

        print("Finish model initialization.")

    def rotation_matrix_2d(self, pseudo_label, inverse=False):
        # print(pseudo_label) (1 * 4) gaze pitch, yaw & head pitch, yaw
        cos = torch.cos(pseudo_label)
        sin = torch.sin(pseudo_label)
        ones = torch.ones_like(cos[:, 0])
        zeros = torch.zeros_like(cos[:, 0])

        matrices_1 = torch.stack([ones, zeros, zeros,
                                  zeros, cos[:, 0], -sin[:, 0],
                                  zeros, sin[:, 0], cos[:, 0]
                                  ], dim=1) # Pitch
        matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                                  zeros, ones, zeros,
                                  -sin[:, 1], zeros, cos[:, 1]
                                  ], dim=1) # Yaw
        matrices_1 = matrices_1.view(-1, 3, 3)
        matrices_2 = matrices_2.view(-1, 3, 3)
        matrices = torch.matmul(matrices_2, matrices_1)
        if inverse:
            matrices = torch.transpose(matrices, 1, 2)
        return matrices

    def rotate(self, embeddings, pseudo_labels, rotate_or_not=None, inverse=False):
        # For gaze or head.
        rotation_matrices = []

        for (dof, _), pseudo_label in zip(self.configuration, pseudo_labels):
            rotation_matrix = self.rotation_matrix_2d(pseudo_label, inverse=inverse)
            rotation_matrices.append(rotation_matrix)

        # rotated gaze and head embeddings.
        rotated_embeddings = [
            torch.matmul(rotation_matrix, embedding)
            if rotation_matrix is not None else embedding
            for embedding, rotation_matrix in zip(embeddings, rotation_matrices)
        ]

        return rotated_embeddings

    def forward(self, data):
        output_dict = OrderedDict()
        losses_dict = OrderedDict()
        
        # Encode input from a
        batch_size = data['image_a'].shape[0]
        data['image_a'] = self.e4e_transforms(data['image_a']) # [batch, 3, 256, 256]
        data['image_b'] = self.e4e_transforms(data['image_b']) # [batch, 3, 256, 256]

        f_a = self.encoder(data['image_a']) # [batch, 9216]
        c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]

        ''' Metric 1'''
        losses_dict['gaze_a'] = losses.gaze_angular_loss(y=data['gaze_a'], y_hat=c_a[-1]) # gaze: -1
        losses_dict['head_a'] = losses.gaze_angular_loss(y=data['head_a'], y_hat=c_a[-2]) # head: -2

        # Direct reconstruction
        fusion_a = self.fusion(f_a).permute(1, 0, 2) # [18, batch, 512]

        # e4e transfer, decode image from G.
        output_rec_a, _ = self.decoder(fusion_a) # [batch, 3, 1024, 1024]
        output_rec_a_256 = F.interpolate(output_rec_a, size=(256, 256), mode='bilinear', align_corners=False) # [batch, 3, 256, 256]

        # Check gaze and head label
        gaze_rec_a, head_rec_a = self.GazeHeadNet_eval(output_rec_a_256) # [batch, 2], [batch, 2]

        # embedding disentanglement error
        idx = 0

        for dof, _ in self.configuration:
            random_angle = (torch.rand(batch_size, dof).to("cuda") - 0.5) * np.pi * 0.2 # [batch, 2]
            random_angle += c_a[idx] # move head or gaze.

            # Check degree of freedom
            transfer_z_random = torch.matmul(self.rotation_matrix_2d(random_angle, False), torch.matmul(
                self.rotation_matrix_2d(c_a[idx], True), z_a[idx])) # [batch, 3, 16]
            
            z_new_a = [item for item in z_a]
            z_new_a[idx] = transfer_z_random # [2, batch, 3, 16] change the rotation angle, 0: change head, 1: change gaze

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_new_a]
            z_new_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
            delta_f_random = self.redirtrans_dp(z_new_a) # [batch, 9216]

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
            z_old_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
            delta_f_a = self.redirtrans_dp(z_old_a) # [batch, 9216]

            transfer_f_random = f_a + delta_f_random - delta_f_a # [batch, 9216]

            fusion_a = self.fusion(transfer_f_random).permute(1, 0, 2) # [batch, 1, 512]
            output_random_a, _ = self.decoder(fusion_a) # [batch, 3, 1024, 1024]
            output_random_a_256 = F.interpolate(output_random_a, size=(256, 256), mode='bilinear', align_corners=False) # [batch, 3, 256, 256]

            # Get the predicted angle.
            gaze_random_a, head_random_a = self.GazeHeadNet_eval(output_random_a_256)

            ''' Metric 2'''
            if idx == config.num_2d_units - 2:  # test head
                losses_dict['head_to_gaze'] = losses.gaze_angular_loss(gaze_rec_a, gaze_random_a)
            if idx == config.num_2d_units - 1:  # test gaze
                losses_dict['gaze_to_head'] = losses.gaze_angular_loss(head_rec_a, head_random_a)

            ''' Metric 3'''
            emb_inversion = self.pretrained_arcface(self.r50_transform(output_rec_a_256))
            emb_target = self.pretrained_arcface(self.r50_transform(data['image_b']))
            emb_ouput_256 = self.pretrained_arcface(self.r50_transform(output_random_a_256))

            losses_dict['id_inversion'] = (1.0 - torch.mean(
                    F.cosine_similarity(emb_inversion,
                                        emb_ouput_256, dim=-1)))
            
            losses_dict['id_target'] = (1.0 - torch.mean(
                    F.cosine_similarity(emb_target,
                                        emb_ouput_256, dim=-1)))
            
            idx += 1

        # Calculate some errors if target image is available
        if 'image_b' in data:
            # redirect with pseudo-labels
            z_b_new_gaze = torch.matmul(self.rotation_matrix_2d(data['gaze_b'], False),
                                          torch.matmul(self.rotation_matrix_2d(c_a[-1], True),
                                                       z_a[-1])) # [batch, 3, 16]
            
            z_b_new_head = torch.matmul(self.rotation_matrix_2d(data['head_b'], False),
                                          torch.matmul(self.rotation_matrix_2d(c_a[-2], True),
                                                       z_a[-2])) # [batch, 3, 16]
            
            transfer_z_b = z_a[:-2]
            transfer_z_b.append(z_b_new_head)
            transfer_z_b.append(z_b_new_gaze)

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
            transfer_z_b = torch.cat(reshaped_tensors, dim=1)
            delta_f_b = self.redirtrans_dp(transfer_z_b)
            # print(f"delta_embeddings_b.shape: {delta_embeddings_b.shape}")

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
            z_a = torch.cat(reshaped_tensors, dim=1)
            delta_f_a = self.redirtrans_dp(z_a)
            # print(f"delta_embeddings_a.shape: {delta_embeddings_a.shape}")

            transfer_f_b = f_a + delta_f_b - delta_f_a
            fusion_f = self.fusion(transfer_f_b).permute(1, 0, 2) # [batch, 1, 512]

            output, _ = self.e4e_net.decoder(fusion_f) # [batch, 3, 1024, 1024]
            output_256 = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False) # [batch, 3, 256, 256]
            output_dict['image_b_hat'] = output_256

            gaze_b_hat, head_b_hat = self.GazeHeadNet_eval(output_dict['image_b_hat'])
            losses_dict['head_redirection'] = losses.gaze_angular_loss(y=data['head_b'], y_hat=head_b_hat)
            losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=data['gaze_b'], y_hat=gaze_b_hat)

            losses_dict['lpips'] = torch.mean(self.lpips(data['image_b'], output_dict['image_b_hat']))
            losses_dict['l1'] = losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat'])

            ###################################################################

            f_a = self.encoder(data['image_a']) # [batch, 9216]
            c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]
            f_b = self.encoder(data['image_b']) # [batch, 9216]
            c_b, z_b = self.redirtrans_p(f_b) # [2, batch, 2], [2, batch, 3, 16]

            # a -> b
            z_n = self.rotate(z_a, c_a, inverse=True) # Transfer to canonical tensor.
            transfer_z_b = self.rotate(z_n, c_b, inverse=False) # Transfer to target tensor.

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
            transfer_z_b = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
            delta_f_b = self.redirtrans_dp(transfer_z_b)

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
            z_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
            delta_f_a = self.redirtrans_dp(z_a)

            transfer_f_b = f_a + delta_f_b - delta_f_a
            fusion_f = self.fusion(transfer_f_b).permute(1, 0, 2) # [batch, 1, 512]

            # e4e transfer, decode image from G.
            output, _ = self.decoder(fusion_f)
            output_256 = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)

            output_dict['image_b_hat_all'] = output_256
            losses_dict['lpips_all'] = torch.mean(self.lpips(data['image_b'], output_dict['image_b_hat_all']))
            losses_dict['l1_all'] = losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat_all'])

        return output_dict, losses_dict

    def optimize(self, data, current_step):
        if config.use_apex:
            from apex import amp
        losses_dict = OrderedDict()

        for param in self.redirtrans_params:
            param.requires_grad = True
        for param in self.fusion_params:
            param.requires_grad = True

        batch_size = data['image_a'].shape[0] # data: dict_keys(['key', 'image_a', 'gaze_a', 'head_a', 'image_b', 'gaze_b', 'head_b'])
        data['image_a'] = self.e4e_transforms(data['image_a'])
        data['image_b'] = self.e4e_transforms(data['image_b'])

        f_a = self.encoder(data['image_a']) # [batch, 9216]
        f_b = self.encoder(data['image_b']) # [batch, 9216]
 
        c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]
        c_b, z_b = self.redirtrans_p(f_b) # [2, batch, 2], [2, batch, 3, 16]

        # a -> b
        z_n = self.rotate(z_a, c_a, inverse=True) # Transfer to canonical tensor.

        transfer_z_b = self.rotate(z_n, c_b, inverse=False) # Transfer to target tensor.

        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
        transfer_z_b = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_b = self.redirtrans_dp(transfer_z_b)

        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
        z_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_a = self.redirtrans_dp(z_a)

        transfer_f_b = f_a + delta_f_b - delta_f_a

        fusion_f = self.fusion(transfer_f_b).permute(1, 0, 2) # [batch, 1, 512]

        # e4e transfer, decode image from G.
        output, _ = self.decoder(fusion_f)
        output_256 = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)

        # Reconstruction image Loss
        losses_dict['l1'] = losses.reconstruction_l1_loss(x=data['image_b'], x_hat=output_256)
        total_loss = losses_dict['l1'] * config.coeff_l1_loss

        # Perceptual Loss
        losses_dict['perceptual'] = torch.mean(self.lpips(data['image_b'], output_256))
        total_loss += losses_dict['perceptual'] * config.coeff_perceptual_loss

        # Gaze and Head attr Loss from Img
        feature_h, gaze_h, head_h = self.GazeHeadNet_train(output_256, True)
        feature_t, gaze_t, head_t = self.GazeHeadNet_train(data['image_b'], True)
        losses_dict['redirection_feature_loss'] = 0
        for i in range(len(feature_h)):
            losses_dict['redirection_feature_loss'] += nn.functional.mse_loss(feature_h[i], feature_t[i].detach())
        total_loss += losses_dict['redirection_feature_loss'] * config.coeff_redirection_feature_loss
        # print(f"redirection_feature_loss: {losses_dict['redirection_feature_loss']}")

        losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=gaze_t.detach(), y_hat=gaze_h)
        total_loss += losses_dict['gaze_redirection'] * config.coeff_redirection_gaze_loss
        # print(f"gaze_redirection_loss: {losses_dict['gaze_redirection']}")

        losses_dict['head_redirection'] = losses.gaze_angular_loss(y=head_t.detach(), y_hat=head_h)
        total_loss += losses_dict['head_redirection'] * config.coeff_redirection_gaze_loss
        # print(f"head_redirection_loss: {losses_dict['head_redirection']}")

        # ID Loss
        emb_target = self.pretrained_arcface(self.r50_transform(data['image_b']))
        emb_ouput_256 = self.pretrained_arcface(self.r50_transform(output_256))
        losses_dict['id'] = (1.0 - torch.mean(
                F.cosine_similarity(emb_target,
                                    emb_ouput_256, dim=-1)))
        total_loss += losses_dict['id'] * config.coeff_id_loss

        # Gaze and Head angle Loss from label
        if not config.semi_supervised:
            losses_dict['gaze_a'] = (losses.gaze_angular_loss(y=data['gaze_a'], y_hat=c_a[-1]) +
                                 losses.gaze_angular_loss(y=data['gaze_b'], y_hat=c_b[-1]))/2
            losses_dict['head_a'] = (losses.gaze_angular_loss(y=data['head_a'], y_hat=c_a[-2]) +
                                 losses.gaze_angular_loss(y=data['head_b'], y_hat=c_b[-2]))/2
        else:
            losses_dict['gaze_a'] = losses.gaze_angular_loss(y=data['gaze_a'], y_hat=c_a[-1])
            losses_dict['head_a'] = losses.gaze_angular_loss(y=data['head_a'], y_hat=c_a[-2])

            losses_dict['gaze_a_unlabeled'] = losses.gaze_angular_loss(y=data['gaze_b'], y_hat=c_b[-1])
            losses_dict['head_a_unlabeled'] = losses.gaze_angular_loss(y=data['head_b'], y_hat=c_b[-2])

        total_loss += (losses_dict['gaze_a'] + losses_dict['head_a']) * config.coeff_gaze_loss

        # Consistency loss
        if config.coeff_embedding_consistency_loss != 0:
            c_a, z_a = self.redirtrans_p(f_a)
            normalized_embeddings_from_a = self.rotate(z_a, c_a, inverse=True)
            c_b, z_b = self.redirtrans_p(f_b)
            normalized_embeddings_from_b = self.rotate(z_b, c_b, inverse=True)

            flattened_normalized_embeddings_from_a = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_a], dim=1)
            flattened_normalized_embeddings_from_b = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_b], dim=1)
            losses_dict['embedding_consistency'] = (1.0 - torch.mean(
                F.cosine_similarity(flattened_normalized_embeddings_from_a,
                                    flattened_normalized_embeddings_from_b, dim=-1)))

            total_loss += losses_dict['embedding_consistency'] * config.coeff_embedding_consistency_loss

        # D-reg and Adv Loss
        losses_dict['discriminator'] = self.train_discriminator(data['image_a'])
        latent = self.forward_latent(data['image_a'])
        losses_dict['adv_reg'] = self.calc_loss(latent)
        total_loss += losses_dict['adv_reg']

        self.redirtrans_optimizer.zero_grad()
        self.fusion_optimizer.zero_grad()
        # self.discriminator_optimizer.zero_grad()

        if config.use_apex:
            # with amp.scale_loss(total_loss, [self.redirtrans_optimizer, self.fusion_optimizer, self.discriminator_optimizer]) as scaled_loss:
            with amp.scale_loss(total_loss, [self.redirtrans_optimizer, self.fusion_optimizer]) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        self.redirtrans_optimizer.step()
        self.fusion_optimizer.step()
        # self.discriminator_optimizer.step()

        return losses_dict, output

    def redirect(self, data):
        output_dict = OrderedDict()
        pseudo_labels_a, embeddings_a = self.encoder(data['image_a'])

        gaze_embedding = torch.matmul(self.rotation_matrix_2d(data['gaze_b_r'], False),
                                  torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-1], True),
                                               embeddings_a[-1]))
        head_embedding = torch.matmul(self.rotation_matrix_2d(data['head_b_r'], False),
                                  torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-2], True),
                                               embeddings_a[-2]))
        embeddings_a_to_b = embeddings_a[:-2]
        embeddings_a_to_b.append(head_embedding)
        embeddings_a_to_b.append(gaze_embedding)
        output_dict['image_b_hat_r'] = self.decoder(embeddings_a_to_b)
        return output_dict

    def clean_up(self):
        self.fusion_optimizer.zero_grad()
        self.redirtrans_optimizer.zero_grad()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ e4e ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.e4e_net.module.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.e4e_net.module.encoder.progressive_stage.value + 1]
    
    def forward_latent(self, x):
        x = x.to("cuda").float()
        _, latent = self.e4e_net.forward(x, return_latents=True)
        return latent
    
    def calc_loss(self, latent):
        loss_dict = {}
        loss = 0.0
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.e4e_net.module.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += config.w_discriminator_lambda * loss_disc

        if config.progressive_steps and self.encoder.progressive_stage.value != 18:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, config.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += config.delta_norm_lambda * total_delta_loss

        loss_dict['loss'] = float(loss)
        return loss


 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean() # SoftPlus is a smooth approximation to the ReLU function
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, x):
        loss_dict = {}
        x = x.to("cuda").float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = config.global_step % config.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = config.r1 / 2 * r1_loss * config.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return float(loss), float(r1_final_loss)

    def validate_discriminator(self, x):
        with torch.no_grad():
            loss_dict = {}
            x = x.to("cuda").float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(config.batch_size, 512, device="cuda")
        real_w = self.e4e_net.module.decoder.get_latent(sample_z)
        fake_w = self.e4e_net.module.encoder(x)
        if config.start_from_latent_avg:
            fake_w = fake_w + self.e4e_net.module.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if config.is_progressive_training:  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if config.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
