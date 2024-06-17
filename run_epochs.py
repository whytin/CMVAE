import torch
from tensorboardX import SummaryWriter
from losses import VMVAE_loss, incomplete_VMVAE_loss, cross_entropy, incomplete_CMVAE_loss, incomplete_CMVAE_loss_stage
import metrics
import numpy as np

def train_eval_classification(exp, epoch, epoch_record):
    model = exp.model
    model.train()
    exp.model = model

    train_set, train_target, test_set, test_target, batch_missing, test_missing = exp.get_dataclassify()
    epoch_loss = 0
    epoch_recons_loss =0
    epoch_cross_entropy = 0
    epoch_kl_loss = 0
    iter_num = 0
    
    for iter, batch in enumerate(train_set):
        logits, recons, mus, log_vars, sample_z, pi = exp.model(batch, batch_missing[iter])
        loss, recons_loss, kl_loss = incomplete_VMVAE_loss(
            batch, recons, mus, log_vars, batch_missing[iter], pi, exp.flags.lamb1, exp.flags.lamb2)
        ce_loss = cross_entropy(logits, train_target[iter])
        loss = loss+ce_loss
        exp.optimizer.zero_grad()
        loss.backward()
        exp.optimizer.step()
        iter_num += 1
        epoch_loss += loss.item()
        epoch_recons_loss += recons_loss.item()
        epoch_cross_entropy += ce_loss.item()
        epoch_kl_loss += kl_loss.item()
    epoch_record['loss'].append(epoch_loss/iter_num)
    epoch_record['recons_loss'].append(epoch_recons_loss/iter_num)
    epoch_record['cross_entropy'].append(epoch_cross_entropy/iter_num)
    epoch_record['kl_loss'].append(epoch_kl_loss/iter_num)
    print('epoch:{}, loss:{}'.format(epoch, epoch_loss/iter_num))

    acc = 0
    with torch.no_grad():
        model.eval()
        logits, _, _, _, _, pi = exp.model(test_set, test_missing)
        _, predicted = torch.max(logits.data, 1)
        acc += (predicted == test_target).sum().item()
        acc = acc/test_target.size(0)
        epoch_record['acc'].append(acc)
        epoch_record['pi'].append(pi.cpu().detach().tolist())
        print('Evaluation classification.....ACC:{}'.format(acc))
    return epoch_record





def train_clustering(exp, epoch, epoch_record):
    model = exp.model
    model.train()
    exp.model = model

    batch_inputs, batch_target, batch_missing = exp.get_trainingdata(exp.flags.full_batch)
    epoch_loss = 0
    epoch_recons_loss = 0
    epoch_kl_loss = 0
    epoch_corr_loss = 0
    iter_num = 0

    for iter, batch in enumerate(batch_inputs):
        if exp.flags.exp_model=='VMVAE_clustering':
            predict, recons, mus, log_vars, sample_z, pi = exp.model(batch, batch_missing[iter])
            #loss, recons_loss, kl_loss = VMVAE_loss(batch, recons, mus, log_vars, pi, exp.flags.lamb1, exp.flags.lamb2)
            loss, recons_loss, kl_loss = incomplete_VMVAE_loss(
                batch, recons, mus, log_vars, batch_missing[iter], pi, exp.flags.lamb1, exp.flags.lamb2)
            corr_loss=0
        elif exp.flags.exp_model=='CMVAE_clustering':
            predict, recons, mus, log_vars, sample_c, pi, view_all_trans, view_recons = exp.model(batch, batch_missing[iter])
            #stage=1
            #if epoch==30:
            #    stage=2
            #elif epoch==60:
            #    stage=3
            #elif epoch==100:
            #    stage=4
            #loss, recons_loss, kl_loss, corr_loss = incomplete_CMVAE_loss_stage(
            #     batch, recons, mus, log_vars, batch_missing[iter], pi, exp.flags.lamb1, exp.flags.lamb2, view_all_trans, view_recons, stage)
            loss, recons_loss, kl_loss, corr_loss = incomplete_CMVAE_loss(
                 batch, recons, mus, log_vars, batch_missing[iter], pi, exp.flags.lamb1, exp.flags.lamb2, view_all_trans, view_recons)
            corr_loss = corr_loss.item()

        exp.optimizer.zero_grad()
        loss.backward()
        exp.optimizer.step()
        iter_num += 1
        epoch_loss += loss.item()
        epoch_recons_loss += recons_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_corr_loss += corr_loss
    epoch_record['loss'].append(epoch_loss/iter_num)
    epoch_record['recons_loss'].append(epoch_recons_loss/iter_num)
    epoch_record['kl_loss'].append(epoch_kl_loss/iter_num)
    epoch_record['corr_loss'].append(epoch_corr_loss/iter_num)

    print('epoch:{}, loss:{}'.format(epoch, epoch_loss/iter_num))
    return epoch_record


def eval_clustering(exp, epoch_record):
    with torch.no_grad():
        model = exp.model
        model.eval()
        exp.model = model

        inputs, target, missing_matrix = exp.get_trainingdata(True)
        if exp.flags.exp_model=='VMVAE_clustering':
            predict, _, _, _, _, pi = exp.model(inputs[0], missing_matrix[0])
        elif exp.flags.exp_model=='CMVAE_clustering':
            predict, _, _, _, _, pi, _, _ = exp.model(inputs[0], missing_matrix[0])
        acc = metrics.acc(target[0], predict)
        nmi = metrics.nmi(target[0].flatten(), predict)
        ari = metrics.ari(target[0].flatten(), predict)
        prt = metrics.purity_score(target[0].flatten(), predict)
        epoch_record['acc'].append(acc)
        epoch_record['nmi'].append(nmi)
        epoch_record['ari'].append(ari)
        epoch_record['prt'].append(prt)
        epoch_record['pi'].append(pi.cpu().detach().tolist())
        print('evaluation.....ACC:{},NMI:{},ARI:{},PURITY:{}'.format(
            acc, nmi, ari, prt))

    return epoch_record


def run_epochs(exp):
    #writer = SummaryWriter(exp.flags.dir_logs)
    if exp.flags.mode == 'clustering':
        epoch_record = {'loss': [], 'recons_loss': [], 'kl_loss': [
        ], 'corr_loss':[], 'acc': [], 'nmi': [], 'ari': [], 'prt': [], 'pi': []}
        print('Clustering training epochs progress: ')
        for epoch in range(exp.flags.epochs):
            train_clustering(exp, epoch, epoch_record)
            eval_clustering(exp, epoch_record)
        epoch_record['best_acc'] = np.max(epoch_record['acc'])
        epoch_record['best_nmi'] = np.max(epoch_record['nmi'])
        epoch_record['best_ari'] = np.max(epoch_record['ari'])
        epoch_record['best_prt'] = np.max(epoch_record['prt'])
    elif exp.flags.mode == 'classification':
        epoch_record = {'loss':[], 'cross_entropy': [], 'recons_loss': [], 'kl_loss': [], 'acc': [], 'pi':[]}
        print('Classification training epochs progress: ')
        for epoch in range(exp.flags.epochs):
            train_eval_classification(exp, epoch, epoch_record)
        epoch_record['best_acc'] = np.max(epoch_record['acc'])
    return epoch_record
