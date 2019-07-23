import tensorflow as tf
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('k', 1, 'when training, we would generate k similar graphs for each one of sampled graphs')
flags.DEFINE_integer('GED_threshold', 5, 'threshold within which 2 graphs are similar')
flags.DEFINE_float('valid_percentage', 0.25, 'percentage of validation set')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('hash_code_len',6,'length of hash code')
flags.DEFINE_integer('batchsize',5,'batch size for training')
flags.DEFINE_string('node_feat_encoder','onehot','How to encode node feature')
flags.DEFINE_string('node_feat_name','type','Name of node feature')
flags.DEFINE_string('node_label_name', 'label', 'Name of node label, none if it\' idx')
flags.DEFINE_string('laplacian','gcn','how to compute laplacian')
flags.DEFINE_float('DSH_loss_m',3,'parameter m for DSH loss')
flags.DEFINE_float('binary_regularizer_weight',0.5,'weight for binary regularizer')
flags.DEFINE_integer('hamming_dist_thres', 2, 'threshold of similar binary codes')
flags.DEFINE_integer('beam_width', 15, 'beam width for BSS_GED')
