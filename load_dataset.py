from sklearn import preprocessing
import numpy as np
import scipy.io as scio


def load_data(tag):
    if tag == 'mnist2view':
        input_dim = [76, 64]
        num_classes = 10
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/MNIST_2views.mat'
        data = scio.loadmat(path)
        X1 = min_max_scaler.fit_transform(data['mode_1']).astype(np.float32)
        X2 = min_max_scaler.fit_transform(data['mode_2']).astype(np.float32)
        inputs = [X1, X2]
        target = data["label"][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim
    elif tag == 'msrc':
        input_dim = [100, 256]
        num_classes = 7
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/MSRC-V1.mat'
        data = scio.loadmat(path)
        X1 = min_max_scaler.fit_transform(data['mode_1']).astype(np.float32)
        X2 = min_max_scaler.fit_transform(data['mode_2']).astype(np.float32)
        inputs = [X1, X2]
        target = data["label"][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim
    elif tag == 'orl':
        input_dim = [4096, 3304]
        num_classes = 40
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/ORL.mat'
        data = scio.loadmat(path)
        X1 = min_max_scaler.fit_transform(data['mode_1']).astype(np.float32)
        X2 = min_max_scaler.fit_transform(data['mode_2']).astype(np.float32)
        inputs = [X1, X2]
        target = data["label"][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'BDGP':
        input_dim = [1750, 79]
        num_classes = 5
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/BDGP.mat'
        data = scio.loadmat(path)
        X1 = min_max_scaler.fit_transform(data['X1']).astype(np.float32)
        X2 = min_max_scaler.fit_transform(data['X2']).astype(np.float32)
        inputs = [X1, X2]
        target = data["Y"][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'UCI':
        input_dim = [240, 76, 216, 47, 64, 6]
        num_classes = 10
        num_views = 6
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/handwritten.mat'
        data = scio.loadmat(path)
        target = data['Y']
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'caltech7':
        input_dim = [48, 40, 254, 1984, 512, 928]
        num_classes = 7
        num_views = 6
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/Caltech101-7.mat'
        data = scio.loadmat(path)
        inputs = []
        target = data['Y']
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'caltech20':
        input_dim = [48, 40, 254, 1984, 512, 928]
        num_classes = 20
        num_views = 6
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/Caltech101-20.mat'
        data = scio.loadmat(path)
        inputs = []
        target = data['Y']
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'scene':
        input_dim = [20, 59, 40]
        num_classes = 15
        num_views = 3
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/Scene-15.mat'
        data = scio.loadmat(path)
        inputs = []
        target = data['Y']
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'landuse':
        input_dim = [20, 59, 40]
        num_classes = 21
        num_views = 3
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/LandUse-21.mat'
        data = scio.loadmat(path)
        inputs = []
        target = data['Y']
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'ORL':
        input_dim = [4096, 3304, 6750]
        num_classes = 40
        num_views = 3
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/ORL_mtv.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(np.transpose(data['X'][0][i])).astype(np.float32))
        target = data['gt']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'NUS':
        input_dim = [65, 226, 145, 74, 129]
        num_classes = 31
        num_views = 5
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/NUSWIDEOBJ_2170.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        target = data['Y']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == "MSRC":
        input_dim = [24, 576, 512, 256, 254]
        num_classes = 7
        num_views = 5
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/MSRC_v1_raw.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['fea'][0][i]).astype(np.float32))
        target = data['gt']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == "reuters":
        input_dim = [2000, 2000, 2000, 2000, 2000]
        num_classes = 6
        num_views = 5
        min_max_scaler = preprocessing.MaxAbsScaler()
        path = '/home/whytin/Research/Datasets/Multi-view/Reuters.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(np.array(data['fea'][0][i]).astype(np.float32))
        target = data['gt']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == "bbc":
        input_dim = [3183, 3203]
        num_classes = 5
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/bbcsport.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['X'][0][i]).astype(np.float32))
        target = data['Y']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == "NH":
        input_dim = [2000, 3304, 6750]
        num_classes = 5
        num_views = 3
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/NH_interval9_mtv.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(np.transpose(data['X'][0][i])).astype(np.float32))
        target = data['gt']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim
    
    elif tag == "Yale":
        input_dim = [4096, 3304, 6750]
        num_classes = 15
        num_views = 3
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/Yale.mat'
        data = scio.loadmat(path)
        inputs = []
        for i in range(num_views):
            inputs.append(min_max_scaler.fit_transform(data['fea'][0][i]).astype(np.float32))
        target = data['gt']
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim
    
    elif tag == 'animal':
        input_dim = [4096,4096]
        num_classes = 50
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/ANIMAL.mat'
        data = scio.loadmat(path)
        inputs = []
        inputs.append(min_max_scaler.fit_transform(data['mode_1']).astype(np.float32))
        inputs.append(min_max_scaler.fit_transform(data['mode_2']).astype(np.float32))
        target = data['label'][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

    elif tag == 'ccv':
        input_dim = [5000, 5000]
        num_classes = 20
        num_views = 2
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        path = '/home/whytin/Research/Datasets/Multi-view/CCV_2views.mat'
        data = scio.loadmat(path)
        inputs = []
        inputs.append(min_max_scaler.fit_transform(data['mode_1']).astype(np.float32))
        inputs.append(min_max_scaler.fit_transform(data['mode_2']).astype(np.float32))
        target = data['label'][0]
        if np.min(target) != 0:
            target -= np.min(target)
        return inputs, target, num_classes, num_views, input_dim

        

def shuffle_data(inputs, target):
    random_seed = np.arange(len(target))
    np.random.shuffle(random_seed)
    new_inputs = []
    for input_data in inputs:
        new_inputs.append(input_data[random_seed])
    new_target= target[random_seed]
    return new_inputs, new_target

    
