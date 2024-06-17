import torch
import torch.nn.functional as F
from torch.nn import Parameter
from .basic_CMVAE import VMVAE, CMVAE
from sklearn.cluster import KMeans
from utils import select_index, incomplete_average_mus

class ClusterAssignment(torch.nn.Module):
    def __init__(self, num_classes, representation_dim, alpha=1.0, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = representation_dim
        self.linear = torch.nn.Linear(representation_dim, num_classes)
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.num_classes, self.latent_dim, dtype=torch.float)
            torch.nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch):
        batch = F.relu(self.linaer(batch))
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class VMVAEClustering(torch.nn.Module):
    def __init__(self, clustering_mode, num_classes, num_views, input_dims, h1, h2, h3, latent_dim, missing_matrix):
        super(VMVAEClustering, self).__init__()
        self.vmvae_model =  VMVAE(num_views, input_dims, h1, h2, h3, latent_dim)
        self.num_classes = num_classes
        #self.missing_matrix = missing_matrix
        self.num_views = num_views
        self.clustering_mode = clustering_mode
        if self.clustering_mode == 'kmeans':
            self.assignment = KMeans(n_clusters=num_classes, n_init=20)
        else:
            self.assignment = ClusterAssignment(num_classes, latent_dim*num_views)

    def forward(self, batch_inputs, missing_matrix):
        recons, mus, log_vars, sample_z, pi = self.vmvae_model(batch_inputs)
        if self.clustering_mode == 'kmeans':
            cluster_features = incomplete_average_mus(mus, missing_matrix)
            #cluster_features = sum(mus)/len(mus)
            #best_ind = torch.argmax(pi, 1)
            #rpts = torch.cat(mus, 1).reshape(batch_inputs[0].size()[0],self.num_views, -1)
            #cluster_features = select_index(rpts, best_ind)
            predict = self.assignment.fit_predict(cluster_features.cpu().detach().numpy())
        else:
            cluster_features = torch.concat(mus, 1)
            predict = self.assignment(cluster_features)
        return predict, recons, mus, log_vars, sample_z, pi


class CMVAEClustering(torch.nn.Module):
    def __init__(self, clustering_mode, num_classes, num_views, input_dims, h1, h2, h3, latent_dim, missing_matrix):
        super(CMVAEClustering, self).__init__()
        self.cmvae_model =  CMVAE(num_views, input_dims, h1, h2, h3, latent_dim)
        self.num_classes = num_classes
        #self.missing_matrix = missing_matrix
        self.num_views = num_views
        self.clustering_mode = clustering_mode
        if self.clustering_mode == 'kmeans':
            self.assignment = KMeans(n_clusters=num_classes, n_init=20)
        else:
            self.assignment = ClusterAssignment(num_classes, latent_dim*num_views)

    def forward(self, batch_inputs, missing_matrix):
        recons, mus, log_vars, sample_z, pi, view_all_trans, view_recons = self.cmvae_model(batch_inputs)
        if self.clustering_mode == 'kmeans':
            cluster_features = incomplete_average_mus(mus, missing_matrix)
            #cluster_features = sum(mus)/len(mus)
            #best_ind = torch.argmax(pi, 1)
            #rpts = torch.cat(mus, 1).reshape(batch_inputs[0].size()[0],self.num_views, -1)
            #cluster_features = select_index(rpts, best_ind)
            predict = self.assignment.fit_predict(cluster_features.cpu().detach().numpy())
        else:
            cluster_features = torch.concat(mus, 1)
            predict = self.assignment(cluster_features)
        return predict, recons, mus, log_vars, sample_z, pi, view_all_trans, view_recons