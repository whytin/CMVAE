from load_dataset import load_data, shuffle_data
from models.CMVAE_clustering import VMVAEClustering, CMVAEClustering
from models.CMVAE_classification import VMVAEClassify
import torch.optim as optim
from utils import batch_extract, get_mask, separate_data
import torch
import numpy as np

class CMVAE_experiment():
    def __init__(self, flags):
        self.flags = flags
        self.missing_matrix=None
        self.inputs, self.target, self.num_classes, self.num_views, self.input_dims = self.set_dataset(
            flags.dataset, flags.is_shuffle)
        self.model = self.set_model()

    def set_dataset(self, dataset_name, is_shuffle):
        inputs, target, num_classes, num_views, input_dims = load_data(
            dataset_name)
        if is_shuffle:
            inputs, target = shuffle_data(inputs, target)
        if self.flags.missing_rate != 0:
            self.missing_matrix=get_mask(num_views, inputs[0].shape[0], self.flags.missing_rate)
            inputs = self.set_incomplete(inputs, self.missing_matrix, num_views)
        return self.data_to_gpu(inputs), target, num_classes, num_views, input_dims
    
    def data_to_gpu(self, inputs):
        new_inputs = []
        for ip in inputs:
            new_inputs.append(torch.from_numpy(ip).to(self.flags.device))
        return new_inputs

    def set_model(self):
        if self.flags.mode == 'clustering':
            if self.flags.exp_model =='VMVAE_clustering':
                model = VMVAEClustering(self.flags.clustering_mode, self.num_classes, self.num_views, self.input_dims, self.flags.h1_dim, self.flags.h2_dim, self.flags.h3_dim, self.num_classes, self.missing_matrix)
            elif self.flags.exp_model == 'CMVAE_clustering':
                model = CMVAEClustering(self.flags.clustering_mode, self.num_classes, self.num_views, self.input_dims, self.flags.h1_dim, self.flags.h2_dim, self.flags.h3_dim, self.num_classes, self.missing_matrix)
        elif self.flags.mode == 'classification':
            model = VMVAEClassify(self.num_classes, self.num_views, self.input_dims, self.flags.h1_dim, self.flags.h2_dim, self.flags.h3_dim, self.num_classes, self.missing_matrix)
        model = model.to(self.flags.device)
        return model
    
    def set_optimizer(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        params = list(self.model.parameters())
        print('num parameters: '+str(total_params))
        optimizer = optim.Adam(params, lr=self.flags.initial_learning_rate, betas=(0.9, 0.999))
        self.optimizer = optimizer
    
    def get_trainingdata(self, full_batch):
        if full_batch:
            return [self.inputs], [self.target], [self.missing_matrix]
        else:
            batch_inputs, batch_target, batch_missing = batch_extract(self.inputs, self.target, self.missing_matrix,  self.flags.batch_size)
            return batch_inputs, batch_target, batch_missing
    
    def get_dataclassify(self):
        train_set, train_target, test_set, test_target, train_missing_matrix, test_missing_matrix = separate_data(self.inputs, torch.from_numpy(self.target.flatten()).cuda(), self.missing_matrix, 0.8)
        batch_inputs, batch_target, batch_missing = batch_extract(train_set, train_target, train_missing_matrix, self.flags.batch_size)
        return batch_inputs, batch_target, test_set, test_target, batch_missing, test_missing_matrix
    
    def set_incomplete(self, inputs, missing_matrix, num_views):
        for v in range(num_views):
            miss = missing_matrix[v]
            inputs[v][miss]=np.zeros((len(miss),inputs[v].shape[-1]))
        return inputs
        


