import torch
import numpy as np


def select_index(tensor3d, tensor1d):
    results = []
    print(tensor1d)
    for sample in tensor3d:
        results.append(sample[tensor1d])
    return torch.cat(results, 1)

def batch_extract(inputs, target, missing_matrix, batch_size):
    batch_inputs=[]
    batch_target = []
    batch_missing = []
    batch_num = int(inputs[0].size()[0]/batch_size)+1
    for b in range(batch_num):
        batch = []
        temp_missing = []
        for i, data in enumerate(inputs):
            temp = missing_matrix[i][missing_matrix[i]>=b*batch_size]
            view_temp = temp[temp<(b+1)*batch_size] - b*batch_size
            temp_missing.append(view_temp)
            batch.append(data[b*batch_size:(b+1)*batch_size, :])
        batch_inputs.append(batch)
        batch_missing.append(temp_missing)
        batch_target.append(target[b*batch_size:(b+1)*batch_size])
    return batch_inputs, batch_target, batch_missing

def get_mask(num_views, num_instances, missing_rate):
    missing_num = int(num_instances*num_views*missing_rate)
    temp_ind = np.array(range(num_instances*num_views))
    ind_matrix = temp_ind.reshape(num_views, num_instances)
    fixed_ind = []
    for i, t in enumerate(np.transpose(ind_matrix)):
        fixed_ind.append(np.random.choice(t, 1))
    leave_ind = np.setdiff1d(temp_ind, fixed_ind)
    missing_ind = np.random.choice(leave_ind, missing_num, replace=False)
    missing_matrix =[]
    for v in range(num_views):
        temp = missing_ind[missing_ind>=v*num_instances]
        view_missing = temp[temp<(v+1)*num_instances]-v*num_instances
        missing_matrix.append(view_missing)
    return missing_matrix

def get_mask1(num_views, num_instances, missing_rate):
    missing_samples = int(num_instances*missing_rate)
    view_missing = int(missing_samples/num_views)
    missing_matrix = []
    missing_samples = []
    temp_box = np.array(range(num_instances))
    bool_box = np.ones(num_instances, dtype=bool)
    for v in range(num_views):
        missing_index = np.random.choice(temp_box,view_missing,replace=False)
        bool_box[missing_index]=False
        missing_matrix.append(np.array(range(num_instances))[bool_box])
        missing_samples.append(missing_index)
        temp_box = np.setdiff1d(temp_box, missing_index)
        bool_box = np.ones(num_instances, dtype=bool)
    return missing_samples

def incomplete_average_mus(mus, missing_matrix):
    for v, mu in enumerate(mus):
        miss = missing_matrix[v]
        mu[miss]=torch.zeros((len(miss), mu.size()[-1])).cuda()
    mu_mean = sum(mus)
    all_miss = []
    for view_miss in missing_matrix:
        all_miss += list(view_miss)
    unique, counts = np.unique(all_miss, return_counts=True)
    counter_dt = dict(zip(unique, counts))
    for i, value in enumerate(mu_mean):
        if i in unique:
            mu_mean[i] /= len(mus)-counter_dt[i]
        else:
            mu_mean[i] /= len(mus)
    return mu_mean

def separate_data(inputs, target, missing_matrix, ratio):   
    num_samples = len(target)
    training_num = int(num_samples*ratio)
    train_set = []
    train_missing_matrix = []
    test_set = []
    test_missing_matrix = []
    for i, ip in enumerate(inputs):
        train_set.append(ip[:training_num])
        train_missing_matrix.append(missing_matrix[i][missing_matrix[i]<training_num])
        test_set.append(ip[training_num:])
        test_missing_matrix.append(missing_matrix[i][missing_matrix[i]>=training_num]-training_num)
    return train_set, target[:training_num], test_set, target[training_num:], train_missing_matrix, test_missing_matrix