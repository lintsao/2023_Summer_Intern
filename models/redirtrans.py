import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class ReDirTrans_P(nn.Module):
    def __init__(self, configuration):
        super(ReDirTrans_P, self).__init__()
        self.configuration = configuration
        self.one_embedding_size = 3 * 16

        self.pseudo = nn.Linear(512 * 18, 96, bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.label_branch = nn.Linear(96, 4, bias=True) # Gaze and head label for pitch and yaw.
        self.tanh = nn.Tanh()

        self.embedding = nn.Linear(512 * 18, 96, bias=True)
        self.branch = nn.Linear(96, self.one_embedding_size * 2, bias=True)

    def forward(self, input):
        flat_pseudo_labels = self.pseudo(input)
        flat_pseudo_labels = self.leaky_relu(flat_pseudo_labels)
        flat_pseudo_labels = self.label_branch(flat_pseudo_labels)
        flat_pseudo_labels = np.pi / 2 * self.tanh(flat_pseudo_labels) # +- 90 degree

        flat_embeddings = self.embedding(input)
        flat_embeddings = self.leaky_relu(flat_embeddings)
        flat_embeddings = self.branch(flat_embeddings)

        # Split the pseudo labels and embeddings up
        pseudo_labels = []
        idx_pl = 0
        for dof, num_feats in self.configuration: # [(2, 16)] do2: 16
            if (idx_pl * dof <= 4):
                pseudo_label = flat_pseudo_labels[:, idx_pl:(idx_pl + dof)]
                pseudo_labels.append(pseudo_label)
                idx_pl += dof

        embeddings = []
        idx_e = 0
        for dof, num_feats in self.configuration: # [(2, 16)] do2: 16
            if (idx_e * dof <= self.one_embedding_size * 2):
                len_embedding = (dof + 1) * num_feats
                flat_embedding = flat_embeddings[:, idx_e:(idx_e + len_embedding)]
                embedding = flat_embedding.reshape(-1, dof + 1, num_feats)
                embeddings.append(embedding)
                idx_e += len_embedding

        return [pseudo_labels, embeddings]

class ReDirTrans_DP(nn.Module):
    def __init__(self, configuration):
        super(ReDirTrans_DP, self).__init__()
        self.one_embedding_size = 3 * 16

        self.fc1 = nn.Linear(self.one_embedding_size * 2, 128, bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 512 * 18, bias=True)

    def forward(self, input):
        x = self.fc1(input)
        x = self.leaky_relu(x)
        x = self.fc2(x)

        return x
    
class Fusion_Layer(nn.Module):
    def __init__(self, num):
        super(Fusion_Layer, self).__init__()
        self.num = num
        self.fc1 = nn.Linear(512 * 18, 512 * self.num, bias=True)

    def forward(self, input):
        x = self.fc1(input)
        x = x.view(x.shape[0], -1, 512) # Should be changed to 512, 18, ?

        return x