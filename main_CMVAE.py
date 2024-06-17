import torch
import os
import json
from flags import parser
from CMVAE_experiment import CMVAE_experiment
from run_epochs import run_epochs

if __name__ == '__main__':
    for lb2 in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    #for lb2 in [0.01]:
        for lb1 in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        #for lb1 in [1]:
            for mr in [0.1,0.2,0.3,0.4,0.5]:
            #for mr in [0.5]:
                for run in range(1, 6):

                    FLAGS = parser.parse_args()

                    FLAGS.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    FLAGS.is_shuffle = True
                    FLAGS.dataset = 'UCI'
                    #FLAGS.exp_model = 'VMVAE_classification'
                    #FLAGS.exp_model = 'VMVAE_clustering'
                    FLAGS.exp_model = 'CMVAE_clustering'
                    FLAGS.missing_rate = mr
                    FLAGS.full_batch = True
                    FLAGS.batch_size = 128
                    FLAGS.lamb1 = lb1
                    FLAGS.lamb2 = lb2
                    FLAGS.epochs = 150
                    FLAGS.initial_learning_rate=0.001
                    FLAGS.mode = 'clustering'
                    #FLAGS.mode = 'classification'
                    FLAGS.save_json_dir = './results/{}/{}'.format(FLAGS.dataset, FLAGS.exp_model)
                    FLAGS.save_json_name = 'mr-{}-lamb1-{}-lamb2-{}-run{}.json'.format(mr, lb1, lb2, run)

                    if not os.path.exists(FLAGS.save_json_dir):
                        os.makedirs(FLAGS.save_json_dir)
                    print('Training dataset: ', FLAGS.dataset)
                    print('Full batch is Ture' if FLAGS.full_batch else 'Batch_size: {}'.format(FLAGS.batch_size))
                    
                    print('Perform representation learning for {}'.format(FLAGS.mode))
                    print('Trade-off parameters lamb1:{}, lamb2:{}')
                    
                    exp =  CMVAE_experiment(FLAGS)
                    exp.set_optimizer()

                    print(FLAGS)

                    results = run_epochs(exp)
                    
                    with open(os.path.join(FLAGS.save_json_dir, FLAGS.save_json_name), 'w') as f:
                        f.write(json.dumps(results))
