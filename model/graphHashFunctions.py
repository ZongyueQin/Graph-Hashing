from layers import GraphConvolution_GCN, Dense, SplitAndMeanPooling
from metrics import binary_regularizer, DSH_loss
from models import Model
import tensorflow as tf

from config import FLAGS

class GraphHash_Naive(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GraphHash_Naive, self).__init__(**kwargs)

        self.inputs = [placeholders['features'], placeholders['support'],
                       placeholders['graph_sizes']]
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hash_code_len
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        self.codes = self.outputs[0] > 0

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += DSH_loss(self.outputs[0], self.placeholders['labels'], 
                              FLAGS.DSH_loss_m)
        self.loss += FLAGS.binary_regularizer_weight*binary_regularizer(self.outputs[0])

    def _accuracy(self):
        pass
        
    def _build(self):
        
        self.layers.append(GraphConvolution_GCN(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_GCN(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(SplitAndMeanPooling(logging=self.logging))
        
        self.layers.append(Dense(input_dim=FLAGS.hidden2,
                                 output_dim=FLAGS.hidden3,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden3,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

