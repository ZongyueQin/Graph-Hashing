import tensorflow as tf
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

#dataset related
dataset = 'FULL_ALCHEMY'

if 'linux' in dataset:
    flags.DEFINE_string('dataset', dataset,'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('node_feat_encoder','constant_1','How to encode node feature')
    flags.DEFINE_string('node_feat_name',None,'Name of node feature')
elif dataset == 'ALCHEMY':
    flags.DEFINE_string('dataset', 'ALCHEMY', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
    flags.DEFINE_string('node_feat_name','a_type','Name of node feature')
elif dataset == 'AIDS': 
    flags.DEFINE_string('dataset', 'AIDS', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
    flags.DEFINE_string('node_feat_name','type','Name of node feature')
elif dataset == 'FULL_ALCHEMY':
    flags.DEFINE_string('dataset', 'FULL_ALCHEMY', 'Dataset string')
    flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
    flags.DEFINE_string('node_feat_name','atom','Name of node feature')
elif dataset == 'BA':
    flags.DEFINE_string('dataset', 'BA','Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
    flags.DEFINE_string('node_feat_name','type','Name of node feature')
else:
    # You should define your own corresponding attributes here
    flags.DEFINE_string('dataset', 'ER', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
    flags.DEFINE_string('node_feat_name','type','Name of node feature')

# flags for sample by proximity
flags.DEFINE_boolean('sample_by_proximity', False, 'if enable sample by proximity')
flags.DEFINE_integer('sample_pool_size', 200, 'Sample Pool Size')
flags.DEFINE_integer('positive_sample_num', 5, 'positive sample number')
flags.DEFINE_integer('update_iter_num', 10, 'Number of iterations to update sample pool') # recommend ecd_batchsize/batchsize


flags.DEFINE_string('bit_weight_type', 'const', 'type of bit weight type, const, log, exp or var')

flags.DEFINE_string('ground_truth_file', 'GT11.txt', 'ground truth file, should be in test directory')
flags.DEFINE_string('node_label_name', 'label', 'Name of node label, none if it\'s idx')
flags.DEFINE_boolean('clip', True, 'clip GED beyond GED_threshold')

# data sample related
flags.DEFINE_integer('max_op', 1, 'maximum operations to generate synthetic graphs')
flags.DEFINE_integer('k', 0, 'when training, we would generate k similar graphs for each one of sampled graphs')
flags.DEFINE_integer('GED_threshold', 11, 'threshold within which 2 graphs are similar')
flags.DEFINE_integer('batchsize',10,'batch size for training')
flags.DEFINE_integer('ecd_batchsize', 100, 'encoding batch size')
flags.DEFINE_string('label_type', 'ged', 'whether training label should be binary or ged')

# loss related
flags.DEFINE_float('exp_a', 0.05, 'a for exp weight')
flags.DEFINE_float('weight_decay', 0.00, 'Weight for L2 loss on embedding matrix.')
#flags.DEFINE_float('DSH_loss_m',24,'parameter m for DSH loss')
flags.DEFINE_float('binary_regularizer_weight',0.2,'weight for binary regularizer')#original 0.2
#flags.DEFINE_float('MAX_BRW', 2, 'maximal binary regularizer weight')
#flags.DEFINE_float('BRW_increase_rate', 5, 'the rate to incrase binary regularizer weight')
flags.DEFINE_float('real_data_loss_weight', 1, 'weight of real data part (loss_1) in MSE_loss')
flags.DEFINE_float('syn_data_loss_weight', 1, 'weight of synthesized data part (loss_2) in MSE_loss')
#flags.DEFINE_float('l1_loss_w',0.0,'weight of l1 loss for codes')
flags.DEFINE_float('code_mse_w', 1, 'weight for code mse loss')
flags.DEFINE_float('emb_mse_w', 10, 'weight for emb mse loss')#orinial 10

# model structure related
#flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('laplacian','gcn','how to compute laplacian')

# layer related
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')

flags.DEFINE_integer('hidden4', 348, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 256, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('embedding_dim', 256, 'dimension of embeddings')

flags.DEFINE_integer('hidden6', 128, 'Number of units in hidden layer 6.')

flags.DEFINE_integer('hash_code_len', 32,'length of hash code')

# training related
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('valid_percentage', 0, 'percentage of validation set')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 15000, 'Number of epochs to train.')
flags.DEFINE_integer('early_stopping_large_range', 500, '')
flags.DEFINE_integer('early_stopping_small_range', 50, '')
flags.DEFINE_integer('early_stopping_check_frequency', 50, '')
flags.DEFINE_float('early_stopping_thres', 0.1, '')
flags.DEFINE_integer('last_n', 5, 'last n loss is used to decide early stopping or not')

# query related
#flags.DEFINE_boolean('fine_grained', True, 'whether use fine grained in range query')
#flags.DEFINE_integer('hamming_dist_thres', 2, 'threshold of similar binary codes')
#flags.DEFINE_integer('top_k', 10, 'how many nearest neighbors to retrieve')

        
#flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('beam_width', 15, 'beam width for BSS_GED')
