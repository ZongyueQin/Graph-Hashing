import tensorflow as tf
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('k', 5, 'when training, we would generate k similar graphs for each one of sampled graphs')
flags.DEFINE_integer('GED_threshold', 8, 'threshold within which 2 graphs are similar')
flags.DEFINE_float('valid_percentage', 0, 'percentage of validation set')
flags.DEFINE_string('dataset', 'AIDS', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')

flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')

flags.DEFINE_integer('hidden4', 348, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 256, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 64, 'Number of units in hidden layer 6.')

flags.DEFINE_integer('hash_code_len',32,'length of hash code')

flags.DEFINE_boolean('fine_grained', True, 'whether use fine grained in range query')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.00, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('DSH_loss_m',24,'parameter m for DSH loss')
flags.DEFINE_float('binary_regularizer_weight',1,'weight for binary regularizer')
flags.DEFINE_float('real_data_loss_weight', 1, 'weight of real data part (loss_1) in MSE_loss')
flags.DEFINE_float('syn_data_loss_weight', 1, 'weight of synthesized data part (loss_2) in MSE_loss')
flags.DEFINE_float('l1_loss_w',0.0,'weight of l1 loss for codes')
        
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batchsize',10,'batch size for training')
flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
flags.DEFINE_string('node_feat_name','type','Name of node feature')
flags.DEFINE_string('node_label_name', 'label', 'Name of node label, none if it\'s idx')
flags.DEFINE_string('laplacian','gcn','how to compute laplacian')
flags.DEFINE_integer('hamming_dist_thres', 2, 'threshold of similar binary codes')
flags.DEFINE_integer('beam_width', 15, 'beam width for BSS_GED')
flags.DEFINE_string('label_type', 'ged', 'whether training label should be binary or ged')
flags.DEFINE_integer('top_k', 10, 'how many nearest neighbors to retrieve')
flags.DEFINE_string('ground_truth_file', 'GT8.txt', 'ground truth file, should be in test directory')
