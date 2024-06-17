import torch
import numpy as np


def VMVAE_loss(inputs, recons, mus, log_vars, pi, lamb1, lamb2):
    recon_loss = 0
    kl_loss = 0
    num_views = len(mus)
    for i in range(num_views):
        for j in range(num_views):
            view_recon_loss = pi[0][i] * \
                torch.sum(
                    0.5*torch.mean((inputs[i]-recons[i+j*num_views])**2, 0))
            recon_loss = recon_loss + view_recon_loss

        kl_loss = kl_loss + \
            pi[0][i]*torch.sum(torch.mean((-0.5*(1+log_vars[i] -
                               mus[i].pow(2)-log_vars[i].exp())), 0))
    loss = lamb1*recon_loss + lamb2*kl_loss
    return loss, recon_loss, kl_loss


def incomplete_VMVAE_loss(inputs, recons, mus, log_vars, missing_matrix, pi, lamb1, lamb2):
    recon_loss = 0
    kl_loss = 0
    num_views = len(mus)
    all_ind = np.array(range(inputs[0].size()[0]))
    for i in range(num_views):
        incom = np.setdiff1d(all_ind, missing_matrix[i])
        for j in range(num_views):
            incom_j = np.setdiff1d(all_ind, missing_matrix[j])
            inter_incom = np.intersect1d(incom, incom_j)
            if np.size(inter_incom)==0:
                view_recon_loss=0
            else:
                view_recon_loss = pi[0][i]*torch.sum(0.5*torch.mean(
                    (inputs[i][inter_incom]-recons[i+j*num_views][inter_incom])**2, 0))
            recon_loss = recon_loss + view_recon_loss

        kl_loss = kl_loss + pi[0][i]*torch.sum(torch.mean(
            (-0.5*(1+log_vars[i][incom]-mus[i][incom].pow(2)-log_vars[i][incom].exp())), 0))
    loss = lamb1*recon_loss + lamb2*kl_loss
    return loss, recon_loss, kl_loss


def incomplete_CMVAE_loss(inputs, recons, mus, log_vars, missing_matrix, pi, lamb1, lamb2, view_all_trans, view_recons):
    recon_loss = 0
    kl_loss = 0
    specific_loss = 0
    num_views = len(mus)
    all_ind = np.array(range(inputs[0].size()[0]))
    for i in range(num_views):
        incom = np.setdiff1d(all_ind, missing_matrix[i])
        specific_loss = specific_loss + pi[0][i]*torch.sum(0.5*torch.mean(
            (inputs[i][incom]-view_recons[i][incom])**2, 0))
        for j in range(num_views):
            incom_j = np.setdiff1d(all_ind, missing_matrix[j])
            inter_incom = np.intersect1d(incom, incom_j)
            view_recon_loss = pi[0][i]*torch.sum(0.5*torch.mean(
                (inputs[i][inter_incom]-recons[i+j*num_views][inter_incom])**2, 0))
            recon_loss = recon_loss + view_recon_loss
        kl_loss = kl_loss + pi[0][i]*torch.sum(torch.mean(
            (-0.5*(1+log_vars[i][incom]-mus[i][incom].pow(2)-log_vars[i][incom].exp())), 0))
    corr_loss = correlation_loss(view_all_trans, missing_matrix, all_ind)
    loss = recon_loss + lamb2*kl_loss + lamb1*corr_loss + specific_loss
    return loss, recon_loss, kl_loss, corr_loss


def incomplete_CMVAE_loss_stage(inputs, recons, mus, log_vars, missing_matrix, pi, lamb1, lamb2, view_all_trans, view_recons, stage):
    recon_loss = 0
    kl_loss = 0
    specific_loss = 0
    num_views = len(mus)
    all_ind = np.array(range(inputs[0].size()[0]))
    for i in range(num_views):
        incom_i = np.setdiff1d(all_ind, missing_matrix[i])
        for j in range(num_views):
            incom_j = np.setdiff1d(all_ind, missing_matrix[j])
            inter_incom = np.intersect1d(incom_i, incom_j)
            specific_loss = specific_loss + pi[0][i]*torch.sum(0.5*torch.mean((inputs[i][inter_incom]-view_recons[i+j*num_views][inter_incom])**2, 0))
            recon_loss = recon_loss + pi[0][i]*torch.sum(0.5*torch.mean((inputs[i][inter_incom]-recons[i+j*num_views][inter_incom])**2, 0))
        kl_loss=kl_loss + pi[0][i]*torch.sum(torch.mean(
            (-0.5*(1+log_vars[i][incom_i]-mus[i][incom_i].pow(2)-log_vars[i][incom_i].exp())), 0))
    corr_loss=correlation_loss(view_all_trans, missing_matrix, all_ind)
    if stage==1:
        loss = specific_loss
    elif stage==2:
        loss = corr_loss
    elif stage==3:
        loss = recon_loss + lamb2*kl_loss 
    elif stage==4:
        loss=recon_loss + lamb2*kl_loss + lamb1*corr_loss + specific_loss
    return loss, recon_loss, kl_loss, corr_loss

def correlation_loss(view_all_trans, missing_matrix, all_ind):
    total_loss=0
    for i in range(len(view_all_trans)):
        incom=np.setdiff1d(all_ind, missing_matrix[i])
        for j in range(len(view_all_trans)):
            if i != j:
                total_loss=total_loss +torch.sum(
                        0.5*torch.mean((view_all_trans[i][i][incom]-view_all_trans[j][i][incom])**2, 0))
    return total_loss

def cross_entropy(logits, target):
    ce_loss=torch.nn.CrossEntropyLoss()
    return ce_loss(logits, target)
