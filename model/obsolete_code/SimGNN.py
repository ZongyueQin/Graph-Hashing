from layers import *
from metrics import *
from models import Model
import tensorflow as tf

from config import FLAGS

class SimGNN(Model):
    def __init__(self, placeholders, input_dim, next_ele, **kwargs):
        super(SimGNN, self).__init__(**kwargs)

        self.graph_embs = []
        self.inputs = [next_ele[0], next_ele[1], next_ele[2]]
        self.labels = next_ele[3]
        self.gen_labels = next_ele[4]
            
        self.input_dim = input_dim
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        
    def _loss(self):
        # Weight decay loss
        self.loss = 0
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.mse_loss, self.pred, self.lab = MSE_Loss_SimGNN(self.pred, self.labels, self.gen_labels)
        self.loss += self.mse_loss       

    def _accuracy(self):
        pass
        
    def _build(self):
        
        conv_layers = []
        conv_layers.append(GraphConvolution_GCN(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=FLAGS.dropout>0,
                                            sparse_inputs=True,
                                            logging=self.logging))

        conv_layers.append(GraphConvolution_GCN(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=FLAGS.dropout>0,
                                            logging=self.logging))

        conv_layers.append(GraphConvolution_GCN(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=FLAGS.dropout>0,
                                            logging=self.logging))



        pool_layers = []
        pool_layers.append(SplitAndAttentionPooling(input_dim=FLAGS.hidden1,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.sigmoid,
                                                    dropout=FLAGS.dropout>0,
                                                    logging=self.logging))
        
        
        pool_layers.append(SplitAndAttentionPooling(input_dim=FLAGS.hidden2,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.sigmoid,
                                                    dropout=FLAGS.dropout>0,
                                                    logging=self.logging))
        


        pool_layers.append(SplitAndAttentionPooling(input_dim=FLAGS.hidden3,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.sigmoid,
                                                    dropout=FLAGS.dropout>0,
                                                    logging=self.logging))
       
        interact_layers = []

        interact_layers.append(SplitIntoPairs(logging=self.logging))
        
        interact_layers.append(NTN(input_dim=FLAGS.hidden1+FLAGS.hidden2+FLAGS.hidden3, 
                                   output_dim=FLAGS.hidden4, 
                                   placeholders=self.placeholders, 
                                   act=tf.nn.relu,
                                   dropout=FLAGS.dropout>0,
                                   bias=True,
                                   logging=self.logging))

        mlp_layers = []


        mlp_layers.append(Dense(input_dim=FLAGS.hidden4,
                                 output_dim=FLAGS.hidden5,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))



        mlp_layers.append(Dense(input_dim=FLAGS.hidden5,
                                 output_dim=FLAGS.hidden6,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

#        mlp_layers.append(Dense(input_dim=FLAGS.hidden5,
#                                 output_dim=FLAGS.hidden6,
#                                 placeholders=self.placeholders,
#                                 act=tf.nn.relu,
#                                 bias=True,
#                                 dropout=FLAGS.dropout>0,
#                                 logging=self.logging))



        mlp_layers.append(Dense(input_dim=FLAGS.hidden6,
                                 output_dim=1,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

        

        
        self.layers = [conv_layers, pool_layers, interact_layers, mlp_layers]

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        self.build_train()
        self.build_feat()
        self.build_pred()

    def build_train(self):

        # Build layer model
        self.activations.append(self.inputs)
        # conv layers
        for conv_layer, pool_layer in zip(self.layers[0], self.layers[1]):
            hidden = conv_layer(self.activations[-1])
            graph_emb = pool_layer(hidden)
            self.graph_embs.append(graph_emb[0])
            self.activations.append(hidden)
        
        
        final_graph_emb = tf.concat(self.graph_embs, axis=1)
        self.activations.append([final_graph_emb, hidden[1], hidden[2]])

        for interact_layer in self.layers[2]:
            hidden = interact_layer(self.activations[-1])
            self.activations.append(hidden)

        for mlp_layer in self.layers[3]:
            hidden = mlp_layer(self.activations[-1])
            self.activations.append(hidden)
                    
        self.outputs = self.activations[-1]
        self.pred = self.outputs[0]
        #self.shape = tf.shape(self.pred)
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
                
    def build_feat(self):
        placeholders = self.placeholders
        self.plhd_inputs = [placeholders['features'], 
                       placeholders['support'],
                       placeholders['graph_sizes']]
        try: 
            self.plhd_inputs[0] = tf.sparse.reorder(self.plhd_inputs[0])
            self.plhd_inputs[1] = tf.sparse.reorder(self.plhd_inputs[1])
        except AttributeError:
            self.plhd_inputs[0] = tf.sparse_reorder(self.plhd_inputs[0])
            self.plhd_inputs[1] = tf.sparse_reorder(self.plhd_inputs[1])
 
        self.plhd_activations = []
        self.plhd_activations.append(self.plhd_inputs)
        self.plhd_graph_embs = []
        # conv layers
        for conv_layer, pool_layer in zip(self.layers[0], self.layers[1]):
            hidden = conv_layer(self.plhd_activations[-1])
            graph_emb = pool_layer(hidden)
            self.plhd_graph_embs.append(graph_emb[0])
            self.plhd_activations.append(hidden)
        
        final_graph_emb = tf.concat(self.plhd_graph_embs, axis=1)
        self.plhd_activations.append([final_graph_emb, hidden[1], hidden[2]])
        
            
        self.plhd_feat = self.plhd_activations[-1][0]

    def build_pred(self):
        placeholders = self.placeholders
        self.pred_inputs = [[placeholders['pred_feat1'],
                             placeholders['pred_feat2']], 
                            None, #placeholders['support'],
                            None] #placeholders['graph_sizes']]
        self.shape=tf.shape(self.pred_inputs[0][1])
        self.pred_activations = []
        self.pred_activations.append(self.pred_inputs)
        for interact_layer in self.layers[2][1:]:
            hidden = interact_layer(self.pred_activations[-1])
            self.pred_activations.append(hidden)

        for simgnn_mlp_layer in self.layers[3]:
            hidden = simgnn_mlp_layer(self.pred_activations[-1])
            self.pred_activations.append(hidden)

        self.plhd_pred = self.pred_activations[-1][0]
        
