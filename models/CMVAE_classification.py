import torch
from .basic_CMVAE import CMVAE, VMVAE
from utils import incomplete_average_mus


class VMVAEClassify(torch.nn.Module):
    def __init__(self, num_classes, num_views, input_dims, h1, h2, h3, latent_dim, missing_matrix):
        super(VMVAEClassify, self).__init__()
        self.vmvae_model = VMVAE(num_views, input_dims, h1, h2, h3, latent_dim)
        self.num_classes = num_classes
        self.missing_matrix = missing_matrix
        self.num_views = num_views
        self.fc = torch.nn.Linear(latent_dim, latent_dim)

    def forward(self, batch_inputs, missing_matrix):
        recons, mus, log_vars, sample_z, pi = self.vmvae_model(batch_inputs)
        classify_features = incomplete_average_mus(mus, missing_matrix)
        logits = self.fc(classify_features)
        return logits, recons, mus, log_vars, sample_z, pi
