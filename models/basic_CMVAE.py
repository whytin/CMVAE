import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, x_in, h1, h2, h3, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(x_in, h1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.linear3 = torch.nn.Linear(h2, h3)
        self._enc_mu = torch.nn.Linear(h3, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(h3, latent_dim)
        self.mu_bn = torch.nn.BatchNorm1d(latent_dim)
        self.mu_bn.weight.requires_grad = False
        torch.nn.init.constant_(self.mu_bn.bias, 0.0)
        self.mu_bn.weight.fill_(0.5)

    def _sample_latent(self, h_enc):
        mu = self._enc_mu(h_enc)
        self.log_var = self._enc_log_sigma(h_enc)
        sigma = torch.exp(self.log_var/2)
        std_z = torch.from_numpy(np.random.normal(
            0, 1, size=sigma.size())).float()
        std_z = std_z.cuda()
        self.z_mean = self.mu_bn(mu)
        self.z_sigma = sigma

        self.z = self.z_mean + self.z_sigma * \
            Variable(std_z, requires_grad=False)

        return self.z, self.z_mean, self.log_var

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        z, z_mean, log_var = self._sample_latent(x)
        return z, z_mean, log_var


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, h3, h2, h1, x_in):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dim, h3)
        self.linear2 = torch.nn.Linear(h3, h2)
        self.linear3 = torch.nn.Linear(h2, h1)
        self.linear4 = torch.nn.Linear(h1, x_in)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Cencoder(torch.nn.Module):
    def __init__(self, x_in, h1, h2, h3, latent_dim):
        super(Cencoder, self).__init__()
        self.linear1 = torch.nn.Linear(x_in, h1)
        self.linear2 = torch.nn.Linear(h1, h2)
        self.linear3 = torch.nn.Linear(h2, h3)
        self._enc_mu = torch.nn.Linear(h3, latent_dim)

    def _sample_latent(self, h_enc, correlation, input_view):
        mu = self._enc_mu(h_enc)
        view_trans, view_concat = correlation(mu, input_view)
        return mu, view_trans, view_concat

    def forward(self, x, correlation, input_view):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mu, view_trans, view_concat = self._sample_latent(x, correlation, input_view)
        return mu, view_trans, view_concat

class Correlation(torch.nn.Module):
    def __init__(self, latent_dim, num_views):
        super(Correlation, self).__init__()
        self.latent_dim = latent_dim
        self.num_views = num_views
        for i in range(num_views):
            for j in range(num_views):
                if i != j:
                    # exec(f'self.linear{i}{j}=torch.nn.Linear(latent_dim, latent_dim)')
                    exec(f'self.linear{i}{j}_1=torch.nn.Linear(latent_dim, 1024)')
                    exec(f'self.linear{i}{j}_2=torch.nn.Linear(1024, latent_dim)')
                    # exec(f'self.linear{i}{j}=torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(latent_dim, latent_dim))')

    def forward(self, view_z, view):
        view_trans = []
        for i in range(self.num_views):
            if i != view:
                exec(f'trans = F.relu(self.linear{view}{i}_1(view_z))')
                exec(f'view_trans.append(F.relu(self.linear{view}{i}_2(trans)))')
                # exec(f'view_trans.append(torch.matmul(view_z, self.linear{view}{i}.weight))')
                # exec(f'view_trans.append(F.relu(self.linear{view}{i}(view_z)))')
            else:
                view_trans.append(view_z)
        view_concat = torch.cat(view_trans, 1)
        return view_trans, view_concat

class Hencoder(torch.nn.Module):
    def __init__(self, latent_dim, num_views):
        super(Hencoder, self).__init__()
        self._henc_mu = torch.nn.Linear(latent_dim*num_views, latent_dim)
        self._henc_log_sigma = torch.nn.Linear(latent_dim*num_views, latent_dim)
        self.mu_bn = torch.nn.BatchNorm1d(latent_dim)
        self.mu_bn.weight.requires_grad = False
        torch.nn.init.constant_(self.mu_bn.bias, 0.0)
        self.mu_bn.weight.fill_(0.5)

    def _sample_latent(self, concat):
        feature_c = self._henc_mu(concat)
        self.log_var = self._henc_log_sigma(concat)
        sigma = torch.exp(self.log_var/2)
        std_c = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        std_c = std_c.cuda()
        self.c_mean = self.mu_bn(feature_c)

        self.c = self.c_mean + sigma * Variable(std_c, requires_grad=False)

        return self.c, self.c_mean, self.log_var

    def forward(self, concat):
        c, c_mean, c_log_var = self._sample_latent(concat)
        return c, c_mean, c_log_var



class CMVAE(torch.nn.Module):
    def __init__(self, num_views, input_dims, h1, h2, h3, latent_dim):
        super(CMVAE, self).__init__()
        self.num_views = num_views
        self.pi = torch.tensor(1/num_views)*torch.ones(1, num_views)
        self.pi = torch.nn.Parameter(F.softmax(self.pi,-1))

        self.correlation = Correlation(latent_dim, num_views)
        
        for i in range(num_views):
            exec(f'self.encoders_{i} = Cencoder(input_dims[i], h1, h2, h3, latent_dim)')
            exec(f'self.decoders_{i} = Decoder(latent_dim, h3, h2, h1, input_dims[i])')
        self.hencoder=Hencoder(latent_dim, self.num_views)
    
    def forward(self, inputs):
        recons = []
        view_recons = []
        mus = []
        view_all_trans = []
        log_vars = []
        sample_c = []
        pi = F.softmax(self.pi,-1)
        for i in range(self.num_views):
            exec(f'z_mu, view_trans, view_concat = self.encoders_{i}(inputs[i], self.correlation, i)')
            exec(f'c, c_mu, c_log_var = self.hencoder(view_concat)')
            exec(f'mus.append(c_mu)')
            exec(f'sample_c.append(c)')
            exec(f'log_vars.append(c_log_var)')
            exec(f'view_all_trans.append(view_trans)')
            exec(f'view_recons.append(self.decoders_{i}(z_mu))')
            for j in range(self.num_views):
                exec(f'recons.append(self.decoders_{j}(c))')
                #exec(f'view_recons.append(self.decoders_{j}(z_mu))')
        return recons, mus, log_vars, sample_c, pi, view_all_trans, view_recons


            


class VMVAE(torch.nn.Module):
    def __init__(self, num_views, input_dims, h1, h2, h3, latent_dim):
        super(VMVAE, self).__init__()

        self.num_views = num_views
        self.pi = torch.tensor(1/num_views)*torch.ones(1, num_views)
        self.pi = torch.nn.Parameter(F.softmax(self.pi,-1))
        
        for i in range(num_views):
            exec(f'self.encoders_{i} = Encoder(input_dims[i], h1, h2, h3, latent_dim)')
            exec(f'self.decoders_{i} = Decoder(latent_dim, h3, h2, h1, input_dims[i])')

    def forward(self, inputs):
        recons = []
        mus = []
        log_vars = []
        sample_z = []
        pi = F.softmax(self.pi,-1)
        for i in range(self.num_views):
            exec(f'z, mu, log_var = self.encoders_{i}(inputs[i])')
            exec(f'mus.append(mu)')
            exec(f'sample_z.append(z)')
            exec(f'log_vars.append(log_var)')
            for j in range(self.num_views):
                exec(f'recons.append(self.decoders_{j}(z))')
        return recons, mus, log_vars, sample_z, pi

    

