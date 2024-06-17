import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='UCI', help='name of the dataset')
parser.add_argument('--is_shuffle', type=bool, default=False, help='Is shuffle the dataset?')
parser.add_argument('--mode', type=str, default='clustering', choices=['clustering', 'classification'], help='choose the model to perform clustering or classification.' )
parser.add_argument('--missing_rate', type=float, default=0.0, choices=[0.0,0.1,0.2,0.3,0.4,0.5], help='missing samples equals to num_views*num_instances*missing_rate')
parser.add_argument('--clustering_mode', type=str, default='kmeans',  help='clustering layers setting.')

parser.add_argument('--full_batch', type=bool, default=True, help='batch_size available when full_batch False')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='training epochs.')
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='training learning rate')

parser.add_argument('--h1_dim', type=int, default=500, help='dimensions of hidden layer 1')
parser.add_argument('--h2_dim', type=int, default=500, help='dimensions of hidden layer 2')
parser.add_argument('--h3_dim', type=int, default=1024, help='dimensions of hidden layer 3')
parser.add_argument('--latent_dim', type=int, default=256, help='dimensions of latent representations, Please reassign the number of classes in training code')

parser.add_argument('--lamb1', type=float, default=1.0, help='trade-off parameters lambda 1')
parser.add_argument('--lamb2', type=float, default=1.0, help='trade-off parameters lambda 2')

parser.add_argument('--dir_logs', type=str, default='./results/logs/', help='tensorboard logs directory')