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

import cv2
from utils import *
import sys

config = DefaultConfig()

class STED(nn.Module):

    def __init__(self):
        super(STED, self).__init__()
        self.configuration = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##########################################################################################################
        # No need in pretrained feature extraction. The feature space is 512 * 18 (18 means the different attributes. We do not know which one is for the gaze and head redirection.)
        self.configuration += ([(2, config.size_2d_unit)] * config.num_2d_units) # [(2, 16)] do2: 16

        self.num_all_pseudo_labels = np.sum([dof for dof, _ in self.configuration]) # 2
        self.num_all_embedding_features = np.sum([(dof + 1) * num_feats for dof, num_feats in self.configuration]) # 3 * 16 = 48
        ##########################################################################################################

        self.e4e_net, self.pretrained_arcface, self.e4e_transforms, self.r50_transform = load_pretrained_model(load_e4e_pretrained=True)

        self.redirtrans_p = ReDirTrans_P(configuration=self.configuration).to(self.device)
        self.redirtrans_dp = ReDirTrans_DP(configuration=self.configuration).to(self.device)
        self.fusion = Fusion_Layer(num=18).to(self.device)

        self.lpips = lpips.LPIPS(net='alex').to(self.device)
        self.GazeHeadNet_train = GazeHeadNet().to(self.device) # vGG16 estimator. 
        self.GazeHeadNet_train.load_state_dict(torch.load(config.pretrained_head_gaze_net_train))
        self.GazeHeadNet_eval = GazeHeadResNet().to(self.device) # ResNet estimator.
        self.GazeHeadNet_eval.load_state_dict(torch.load(config.pretrained_head_gaze_net_eval))
        self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
        self.real_w_pool = LatentCodesPool(config.w_pool_size)
        self.fake_w_pool = LatentCodesPool(config.w_pool_size)

        self.pretrained_arcface_params = []
        self.redirtrans_params = []
        self.fusion_params = []
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
        for param in self.lpips_params:
            param.requires_grad = False
        for param in self.GazeHeadNet_train_params:
            param.requires_grad = False
        for param in self.GazeHeadNet_eval_params:
            param.requires_grad = False
        
        self.redirtrans_optimizer = optim.Adam(self.redirtrans_params, lr=config.lr, weight_decay=config.l2_reg)
        self.fusion_optimizer = optim.Adam(self.fusion_params, lr=config.lr, weight_decay=config.l2_reg)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator_params, lr=config.w_discriminator_lr)
        self.optimizers = [self.fusion_optimizer, self.redirtrans_optimizer, self.discriminator_optimizer]

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

    def rotate(self, embeddings, pseudo_labels, inverse=False):
        # For gaze or head.
        rotation_matrices = []

        for (dof, _), pseudo_label in zip(self.configuration, pseudo_labels):
            # print(pseudo_labels)
            rotation_matrix = self.rotation_matrix_2d(pseudo_label, inverse=inverse)
            # print(rotation_matrix)
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

        data['image_a_inv'], _ = self.e4e_net(data['image_a'].float(), randomize_noise=False, return_latents=True)
        output_dict['image_a_inv'] = data['image_a_inv']
        gaze_a_inv, head_a_inv = self.GazeHeadNet_eval(data['image_a_inv'])
        
        
        f_a = self.e4e_net.module.encoder(data['image_a']).contiguous().view(batch_size, -1) # [batch, 9216]
        c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]

        ''' Metric 1: Angular Loss with gt labels.'''
        losses_dict['gaze_a'] = losses.gaze_angular_loss(y=gaze_a_inv, y_hat=c_a[-1]) # gaze: -1
        losses_dict['head_a'] = losses.gaze_angular_loss(y=head_a_inv, y_hat=c_a[-2]) # head: -2

        # Direct reconstruction
        fusion_a = f_a.contiguous().view(batch_size, -1, 512)
        fusion_a = self.e4e_add_latent(fusion_a)

        # e4e transfer, decode image from G.
        output_rec_a, _ = self.e4e_net.module.decoder([fusion_a], input_is_latent=True, randomize_noise=False, return_latents=True) # [batch, 3, 1024, 1024]
        output_rec_a_256 = self.e4e_net.module.face_pool(output_rec_a) # [batch, 3, 256, 256]

        # Check gaze and head label
        gaze_rec_a, head_rec_a = self.GazeHeadNet_eval(output_rec_a_256) # [batch, 2], [batch, 2]

        # embedding disentanglement error
        idx = 0

        for dof, _ in self.configuration:
            if dof == idx:
                break

            random_angle = (torch.rand(batch_size, dof).to(self.device) - 0.5) * np.pi * 0.2 # [batch, 2]
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
            fusion_a = transfer_f_random.contiguous().view(batch_size, -1, 512)
            fusion_a = self.e4e_add_latent(fusion_a)

            output_random_a, _ = self.e4e_net.module.decoder([fusion_a], input_is_latent=True, randomize_noise=False, return_latents=True) # [batch, 3, 1024, 1024]
            output_random_a_256= self.e4e_net.module.face_pool(output_random_a) # [batch, 3, 256, 256]

            # Get the predicted angle.
            gaze_random_a, head_random_a = self.GazeHeadNet_eval(output_random_a_256)

            ''' Metric 2: Disentanglement error'''
            if idx == config.num_2d_units - 2:  # test head
                losses_dict['gaze_consistency'] = losses.gaze_angular_loss(gaze_rec_a, gaze_random_a)
            if idx == config.num_2d_units - 1:  # test gaze
                losses_dict['head_consistency'] = losses.gaze_angular_loss(head_rec_a, head_random_a)

            idx += 1

        # Calculate some errors if target image is available
        if 'image_b' in data:

            # Use GT as label
            data['image_b_inv'], _ = self.e4e_net(data['image_b'].float(), randomize_noise=False, return_latents=True)
            output_dict['image_b_inv'] = data['image_b_inv']
            gaze_b_inv, head_b_inv = self.GazeHeadNet_eval(data['image_b_inv'])
            
            # redirect with gt-labels
            z_n = self.rotate(z_a, c_a, inverse=True)
            c_b_gt = [head_b_inv, gaze_b_inv]

            # Step 4.2: Transfer to target tensor.
            transfer_z_b = self.rotate(z_n, c_b_gt, inverse=False) # Transfer to target tensor.

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
            transfer_z_b = torch.cat(reshaped_tensors, dim=1)
            delta_f_b = self.redirtrans_dp(transfer_z_b)
            # print(f"delta_embeddings_b.shape: {delta_embeddings_b.shape}")

            reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
            z_a = torch.cat(reshaped_tensors, dim=1)
            delta_f_a = self.redirtrans_dp(z_a)
            # print(f"delta_embeddings_a.shape: {delta_embeddings_a.shape}")

            transfer_f_b = f_a + delta_f_b - delta_f_a
            fusion_f = transfer_f_b.contiguous().view(batch_size, -1, 512) # [batch, 9216]
            fusion_f = self.e4e_add_latent(fusion_f)

            output, _ = self.e4e_net.module.decoder([fusion_f], input_is_latent=True, randomize_noise=False, return_latents=True) # [batch, 3, 1024, 1024]
            output_256 = self.e4e_net.module.face_pool(output) # [batch, 3, 256, 256]
            output_dict['image_b_hat'] = output_256

            gaze_b_hat, head_b_hat = self.GazeHeadNet_eval(output_dict['image_b_hat'])
            losses_dict['head_redirection'] = losses.gaze_angular_loss(y=gaze_b_inv, y_hat=head_b_hat)
            losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=head_b_inv, y_hat=gaze_b_hat)

            losses_dict['lpips'] = torch.mean(self.lpips(data['image_b_inv'], output_dict['image_b_hat']))
            losses_dict['l2'] = losses.reconstruction_l2_loss(data['image_b_inv'], output_dict['image_b_hat'])

            ''' Metric 3: ID'''
            emb_inversion = self.pretrained_arcface(self.r50_transform(data['image_b_inv']))
            emb_target = self.pretrained_arcface(self.r50_transform(data['image_b']))
            emb_ouput_256 = self.pretrained_arcface(self.r50_transform(output_256))

            losses_dict['id_target'] = (1.0 - torch.mean(F.cosine_similarity(emb_target, emb_ouput_256, dim=-1)))
            losses_dict['id_inversion'] = (1.0 - torch.mean(F.cosine_similarity(emb_inversion, emb_ouput_256, dim=-1)))


            ###################################################################
            # Use Pseudo as label

            f_a = self.e4e_net.module.encoder(data['image_a_inv']).contiguous().view(batch_size, -1) # [batch, 9216]
            c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]
            f_b = self.e4e_net.module.encoder(data['image_b_inv']).contiguous().view(batch_size, -1) # [batch, 9216]
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
            fusion_f = transfer_f_b.contiguous().view(batch_size, -1, 512) # [batch, 9216]
            fusion_f = self.e4e_add_latent(fusion_f)

            # e4e transfer, decode image from G.
            output, _ = self.e4e_net.module.decoder([fusion_f], input_is_latent=True, randomize_noise=False, return_latents=True) # [batch, 3, 1024, 1024]
            output_256 = self.e4e_net.module.face_pool(output) # [batch, 3, 256, 256]

            emb_ouput_256 = self.pretrained_arcface(self.r50_transform(output_256))

            output_dict['image_b_hat_all'] = output_256
            losses_dict['lpips_all'] = torch.mean(self.lpips(data['image_b_inv'], output_dict['image_b_hat_all']))
            losses_dict['l2_all'] = losses.reconstruction_l2_loss(data['image_b_inv'], output_dict['image_b_hat_all'])

            losses_dict['id_target'] = (1.0 - torch.mean(F.cosine_similarity(emb_target, emb_ouput_256, dim=-1)))
            losses_dict['id_inversion'] = (1.0 - torch.mean(F.cosine_similarity(emb_inversion, emb_ouput_256, dim=-1)))

        return output_dict, losses_dict

    def optimize(self, data, current_step):
        losses_dict = OrderedDict()

        for param in self.redirtrans_params:
            param.requires_grad = True
        # for param in self.fusion_params:
        #     param.requires_grad = True
        for param in self.discriminator_params:
            param.requires_grad = True

        batch_size = data['image_a'].shape[0] # data: dict_keys(['key', 'image_a', 'gaze_a', 'head_a', 'image_b', 'gaze_b', 'head_b'])
        save_images(data['image_a'][0], "input_a.png", fromTransformTensor=True)
        save_images(data['image_b'][0], "input_b.png", fromTransformTensor=True)
        
        # Step 1: Image transformation.
        data['image_a_inv'], _ = self.e4e_net(data['image_a'].float(), randomize_noise=False, return_latents=True)
        data['image_b_inv'], _ = self.e4e_net(data['image_b'].float(), randomize_noise=False, return_latents=True)

        # Step 2: Get the pretrained feature and head / gaze label
        save_images(data['image_a_inv'][0], "input_a_inv.png", fromTransformTensor=True)
        save_images(data['image_b_inv'][0], "input_b_inv.png", fromTransformTensor=True)

        f_a = self.e4e_net.module.encoder(data['image_a_inv']).contiguous().view(batch_size, -1) # [batch, 9216]
        f_b = self.e4e_net.module.encoder(data['image_b_inv']).contiguous().view(batch_size, -1) # [batch, 9216]

        _, gaze_a, head_a = self.GazeHeadNet_train(data['image_a_inv'], True)
        _, gaze_b, head_b = self.GazeHeadNet_train(data['image_a_inv'], True)

        # Step 3: Get the condition and the embedding
        c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]
        c_b, z_b = self.redirtrans_p(f_b) # [2, batch, 2], [2, batch, 3, 16]
        
        # Step 4: Transfer.
        # Step 4.1: Transfer to canonical tensor.
        z_n = self.rotate(z_a, c_a, inverse=True)

        # Step 4.2: Transfer to target tensor.
        transfer_z_b = self.rotate(z_n, c_b, inverse=False) # Transfer to target tensor.

        # Step 5: get the delta embedding.
        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
        transfer_z_b = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_b = self.redirtrans_dp(transfer_z_b) # [batch, 9216]

        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
        z_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_a = self.redirtrans_dp(z_a) # [batch, 9216]

        # Step 6: Feature editting and calculate the weight.
        transfer_f_b = f_a + delta_f_b - delta_f_a
        # fusion_f = self.fusion(f_a) # [batch, layers, 512]
        fusion_f = transfer_f_b.contiguous().view(batch_size, -1, 512) # [batch, 9216]

        # e4e adding latent.
        fusion_f = self.e4e_add_latent(fusion_f)

        # Step 7: e4e transfer, decode image from G.
        output, _ = self.e4e_net.module.decoder([fusion_f], input_is_latent=True, randomize_noise=False, return_latents=True)
        output_256 = self.e4e_net.module.face_pool(output)
        save_images(output_256[0], "output_image.png", fromTransformTensor=True)

        ############################## Loss ##############################
        total_loss = 0.0
        # Loss 1: Reconstruction image Loss
        # losses_dict['l2'] = losses.reconstruction_l2_loss(x=data['image_b'], x_hat=output_256)
        losses_dict['l2'] = losses.reconstruction_l2_loss(x=data['image_b_inv'], x_hat=output_256)
        total_loss += losses_dict['l2'] * config.coeff_l2_loss

        # Loss 2: Perceptual Loss
        # losses_dict['perceptual'] = torch.mean(self.lpips(data['image_b'], output_256))
        losses_dict['perceptual'] = torch.mean(self.lpips(data['image_b_inv'], output_256))
        total_loss += losses_dict['perceptual'] * config.coeff_perceptual_loss

        # Loss 3: Gaze and Head attr Loss from Img
        feature_h, gaze_output, head_output = self.GazeHeadNet_train(output_256, True)
        # feature_t, gaze_t, head_t = self.GazeHeadNet_train(data['image_b'], True)
        # feature_t, gaze_t, head_t = self.GazeHeadNet_train(data['image_b_inv'], True)
        # losses_dict['redirection_feature_loss'] = 0
        # for i in range(len(feature_h)):
        #     losses_dict['redirection_feature_loss'] += nn.functional.mse_loss(feature_h[i], feature_t[i].detach())
        # total_loss += losses_dict['redirection_feature_loss'] * config.coeff_redirection_feature_loss
        # print(f"redirection_feature_loss: {losses_dict['redirection_feature_loss']}")

        losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=gaze_output.detach(), y_hat=gaze_b)
        total_loss += losses_dict['gaze_redirection'] * config.coeff_redirection_gaze_loss
        # print(f"gaze_redirection_loss: {losses_dict['gaze_redirection']}")

        losses_dict['head_redirection'] = losses.gaze_angular_loss(y=head_output.detach(), y_hat=head_b)
        total_loss += losses_dict['head_redirection'] * config.coeff_redirection_head_loss
        # print(f"head_redirection_loss: {losses_dict['head_redirection']}")

        # Loss 4: ID Loss
        emb_target = self.pretrained_arcface(self.r50_transform(data['image_a_inv']))
        emb_ouput_256 = self.pretrained_arcface(self.r50_transform(output_256))
        losses_dict['id'] = (1.0 - torch.mean(F.cosine_similarity(emb_target, emb_ouput_256, dim=-1)))
        total_loss += losses_dict['id'] * config.coeff_id_loss

        # Loss 5: Gaze and Head angle Loss from label
        if not config.semi_supervised:
            losses_dict['gaze_a'] = (losses.gaze_angular_loss(y=gaze_a, y_hat=c_a[-1]) +
                                 losses.gaze_angular_loss(y=head_b, y_hat=c_b[-1]))/2
            losses_dict['head_a'] = (losses.gaze_angular_loss(y=head_a, y_hat=c_a[-2]) +
                                 losses.gaze_angular_loss(y=head_b, y_hat=c_b[-2]))/2
        else:
            losses_dict['gaze_a'] = losses.gaze_angular_loss(y=gaze_a, y_hat=c_a[-1])
            losses_dict['head_a'] = losses.gaze_angular_loss(y=head_a, y_hat=c_a[-2])

            losses_dict['gaze_a_unlabeled'] = losses.gaze_angular_loss(y=gaze_b, y_hat=c_b[-1])
            losses_dict['head_a_unlabeled'] = losses.gaze_angular_loss(y=head_b, y_hat=c_b[-2])

        total_loss += (losses_dict['gaze_a'] + losses_dict['head_a']) * config.coeff_gaze_head_label_loss

        # Loss 6: Consistency loss
        if config.coeff_embedding_consistency_loss != 0:
            c_a, z_a = self.redirtrans_p(f_a)
            normalized_embeddings_from_a = self.rotate(z_a, c_a, inverse=True)
            c_b, z_b = self.redirtrans_p(f_b)
            normalized_embeddings_from_b = self.rotate(z_b, c_b, inverse=True)

            flattened_normalized_embeddings_from_a = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_a], dim=1)
            flattened_normalized_embeddings_from_b = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_b], dim=1)
            losses_dict['embedding_consistency'] = (1.0 - torch.mean(F.cosine_similarity(flattened_normalized_embeddings_from_a, flattened_normalized_embeddings_from_b, dim=-1)))

            total_loss += losses_dict['embedding_consistency'] * config.coeff_embedding_consistency_loss

        # Loss 7: D-reg and Adv Loss
        # losses_dict['discriminator_loss'], losses_dict['discriminator_r1_loss'] = self.train_discriminator(data['image_a'], fusion_f)

        # latent = self.forward_latent(data['image_a'])
        # losses_dict['adv_reg'] = self.calc_loss(latent)
        # total_loss += losses_dict['adv_reg']

        # Step 8: Optimize.
        self.redirtrans_optimizer.zero_grad()
        # self.fusion_optimizer.zero_grad()
        # self.discriminator_optimizer.zero_grad()

        total_loss.backward()

        self.redirtrans_optimizer.step()
        # self.fusion_optimizer.step()
        # self.discriminator_optimizer.step()

        output = None

        return losses_dict, output
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ e4e ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def e4e_add_latent(self, latent, flag=True):
        # e4e adding latent.
        if flag:
            if self.e4e_net.module.opts.start_from_latent_avg:
                if latent.ndim == 2:
                    latent = latent + self.e4e_net.module.latent_avg.repeat(latent.shape[0], 1, 1)[:, 0, :].to(self.device)
                else:
                    latent = latent + self.e4e_net.module.latent_avg.repeat(latent.shape[0], 1, 1).to(self.device)
        else:
            if self.e4e_net.opts.start_from_latent_avg:
                if latent.ndim == 2:
                    latent = latent + self.e4e_net.latent_avg.repeat(latent.shape[0], 1, 1)[:, 0, :].to(self.device)
                else:
                    latent = latent + self.e4e_net.latent_avg.repeat(latent.shape[0], 1, 1).to(self.device)          
        return latent


    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.e4e_net.module.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.e4e_net.module.encoder.progressive_stage.value + 1]
    
    def forward_latent(self, x):
        x = x.to(self.device).float()
        _, latent = self.e4e_net.forward(x, return_latents=True)
        return latent
    
    def calc_loss(self, latent):
        loss_dict = {}
        loss = 0.0
        if config.is_training_discriminator:  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if config.is_progressive_training else \
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

    def train_discriminator(self, x, embeddings):
        loss_dict = {}

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x, embeddings)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = torch.tensor([float(loss)])

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

            self.discriminator_optimizer.zero_grad()
            r1_final_loss = config.r1 / 2 * r1_loss * config.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = torch.tensor([float(r1_final_loss)])

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict['discriminator_loss'], loss_dict['discriminator_r1_loss']

    def validate_discriminator(self, x):
        with torch.no_grad():
            loss_dict = {}
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x, embeddings):
        sample_z = torch.randn(18, config.batch_size, 512, device=self.device)
        real_w = self.e4e_net.module.decoder.get_latent(sample_z)
        # print(f"real {real_w.shape}")
        # fake_w = self.e4e_net.module.encoder(x)
        fake_w = embeddings
        # print(f"fake {fake_w.shape}")
        # if config.start_from_latent_avg:
        #     fake_w = fake_w.to(device) + self.e4e_net.module.latent_avg.repeat(fake_w.shape[0], 1, 1).to(device)
        # if config.is_progressive_training:  # When progressive training, feed only unique w's
        #     dims_to_discriminate = self.get_dims_to_discriminate()
        #     fake_w = fake_w[:, dims_to_discriminate, :]
        # if config.use_w_pool:
        #     real_w = self.real_w_pool.query(real_w)
        #     fake_w = self.fake_w_pool.query(fake_w)
        # if fake_w.ndim == 3:
        #     fake_w = fake_w[:, 0, :]
        return real_w.contiguous().view(-1, 512 * 18), fake_w.contiguous().view(-1, 512 * 18)


    def redirect(self, data):
        losses_dict = OrderedDict()
        output_dict = OrderedDict()
        label_dict = OrderedDict()

        batch_size = data['image_a'].shape[0]

        data['image_a_inv'], _ = self.e4e_net(data['image_a'].float(), randomize_noise=False, return_latents=True)

        gaze_a_inv, head_a_inv = self.GazeHeadNet_eval(data['image_a_inv'])

        # Get the pretrained feature
        f_a = self.e4e_net.encoder(data['image_a_inv']).contiguous().view(batch_size, -1) # [batch, 9216]

        # Get the condition and the embedding
        c_a, z_a = self.redirtrans_p(f_a) # [2, batch, 2], [2, batch, 3, 16]
        
        # Transfer.
        # redirect with gt-labels
        z_n = self.rotate(z_a, c_a, inverse=True)
        c_b = [data['head_b_r'], data['gaze_b_r']]

        # Transfer to target tensor.
        transfer_z_b = self.rotate(z_n, c_b, inverse=False) # Transfer to target tensor.
        
        # Get the delta embedding.
        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in transfer_z_b]
        transfer_z_b = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_b = self.redirtrans_dp(transfer_z_b) # [batch, 9216]

        reshaped_tensors = [tensor.view(batch_size, -1) for tensor in z_a]
        z_a = torch.cat((reshaped_tensors[0], reshaped_tensors[1]), dim=1)
        delta_f_a = self.redirtrans_dp(z_a) # [batch, 9216]

        # Feature editting and calculate the weight.
        transfer_f_b = f_a + delta_f_b - delta_f_a
        fusion_f = transfer_f_b.contiguous().view(batch_size, -1, 512) # [batch, 9216]
        
        # e4e adding latent.
        fusion_f = self.e4e_add_latent(fusion_f, flag=False)

        # e4e transfer, decode image from G.
        output, _ = self.e4e_net.decoder([fusion_f], input_is_latent=True, randomize_noise=False, return_latents=True)
        output_256 = self.e4e_net.face_pool(output)

        gaze_output, head_output = self.GazeHeadNet_eval(output_256)

        label_dict['head_a_inv'] = [torch.rad2deg(head_a_inv[0][0]), torch.rad2deg(head_a_inv[0][1])]
        label_dict['gaze_a_inv'] = [torch.rad2deg(gaze_a_inv[0][0]), torch.rad2deg(gaze_a_inv[0][1])]
        label_dict['head_target'] = [torch.rad2deg(data['head_b_r'][0][0]), torch.rad2deg(data['head_b_r'][0][1])]
        label_dict['gaze_target'] = [torch.rad2deg(data['gaze_b_r'][0][0]), torch.rad2deg(data['gaze_b_r'][0][1])]
        label_dict['head_output'] = [torch.rad2deg(head_output[0][0]), torch.rad2deg(head_output[0][1])]
        label_dict['gaze_output'] = [torch.rad2deg(gaze_output[0][0]), torch.rad2deg(gaze_output[0][1])]

        losses_dict['gaze_redirect'] = losses.gaze_angular_loss(y=data['gaze_b_r'], y_hat=gaze_output) # gaze: -1
        losses_dict['head_redirect'] = losses.gaze_angular_loss(y=data['head_b_r'], y_hat=head_output) # head: -2

        output_dict['image_a_inv'] = data['image_a_inv']
        output_dict["output_1024"] = output
        output_dict["output_256"] = output_256

        return output_dict, losses_dict, label_dict

    def clean_up(self):
        self.fusion_optimizer.zero_grad()
        self.redirtrans_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
