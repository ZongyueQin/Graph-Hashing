from layers import *
from metrics import *
from models import Model
import tensorflow as tf

from config import FLAGS

class GraphHash_Naive(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GraphHash_Naive, self).__init__(**kwargs)

        self.graph_embs = []
        self.inputs = [placeholders['features'], placeholders['support'],
                       placeholders['graph_sizes']]
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hash_code_len
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        self.codes = self.outputs[0] > placeholders['thres']

    def _loss(self):
        # Weight decay loss
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += DSH_loss(self.outputs[0], self.placeholders['labels'], 
                              FLAGS.DSH_loss_m)
        self.loss += FLAGS.binary_regularizer_weight*binary_regularizer(self.outputs[0])

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
       

        mlp_layers = []
        mlp_layers.append(Dense(input_dim=FLAGS.hidden1+FLAGS.hidden2+FLAGS.hidden3,
                                 output_dim=FLAGS.hidden4,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

        mlp_layers.append(Dense(input_dim=FLAGS.hidden4,
                                 output_dim=FLAGS.hidden5,
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



        mlp_layers.append(Dense(input_dim=FLAGS.hidden5,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

        

        
        self.layers = [conv_layers, pool_layers, mlp_layers]

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

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
        
        for mlp_layer in self.layers[2]:
            hidden = mlp_layer(self.activations[-1])
            
            self.activations.append(hidden)
            
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)





class GraphHash_Rank(GraphHash_Naive):
    def __init__(self, placeholders, input_dim, next_ele, **kwargs):
        super(GraphHash_Naive, self).__init__(**kwargs)

        self.graph_embs = []
        self.inputs = [next_ele[0], next_ele[1], next_ele[2]]
        self.labels = next_ele[3]
        self.gen_labels = None
        if FLAGS.label_type == 'ged':
            self.gen_labels = next_ele[4]
            
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hash_code_len
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        self.build_encode()
        
    def build_encode(self):
        placeholders = self.placeholders
        self.ecd_inputs = [placeholders['features'], 
                       placeholders['support'],
                       placeholders['graph_sizes']]
        try: 
            self.ecd_inputs[0] = tf.sparse.reorder(self.ecd_inputs[0])
            self.ecd_inputs[1] = tf.sparse.reorder(self.ecd_inputs[1])
        except AttributeError:
            self.ecd_inputs[0] = tf.sparse_reorder(self.ecd_inputs[0])
            self.ecd_inputs[1] = tf.sparse_reorder(self.ecd_inputs[1])
 
        self.ecd_activations = []
        self.ecd_activations.append(self.ecd_inputs)
        self.ecd_graph_embs = []
        # conv layers
        for conv_layer, pool_layer in zip(self.layers[0], self.layers[1]):
            hidden = conv_layer(self.ecd_activations[-1])
            graph_emb = pool_layer(hidden)
            self.ecd_graph_embs.append(graph_emb[0])
            self.ecd_activations.append(hidden)
        
        final_graph_emb = tf.concat(self.ecd_graph_embs, axis=1)
        self.ecd_activations.append([final_graph_emb, hidden[1], hidden[2]])
        
        for mlp_layer in self.layers[2]:
            hidden = mlp_layer(self.ecd_activations[-1])
            self.ecd_activations.append(hidden)
            
        self.encode_outputs = self.ecd_activations[-1]
        self.codes = self.encode_outputs[0] > placeholders['thres']

    def _loss(self):
        # Weight decay loss
        self.loss = 0
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss = self.loss+FLAGS.weight_decay * tf.nn.l2_loss(var)

        mse_loss, pred, lab = MSE_Loss(self.outputs[0], 
                              self.labels, 
                              self.gen_labels)

        self.loss = self.loss + l1_loss(self.outputs[0])

        self.loss = self.loss + mse_loss
        self.mse_loss = mse_loss
        self.pred = pred
        self.lab = lab

class GraphHash_Rank_Reg(GraphHash_Rank):
    def __init__(self, placeholders, input_dim, next_ele, **kwargs):
        super(GraphHash_Naive, self).__init__(**kwargs)

        self.graph_embs = []
        self.inputs = [next_ele[0], next_ele[1], next_ele[2]]
        self.labels = next_ele[3]
        self.gen_labels = None
        if FLAGS.label_type == 'ged':
            self.gen_labels = next_ele[4]
            
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.hash_code_len
        self.placeholders = placeholders

        if 'binary_reg_w' in placeholders.keys():
            print('auto increasing binary regularizer weight')
            self.binary_reg_w = placeholders['binary_reg_w']
        else:
            self.binary_reg_w = FLAGS.binary_regularizer_weight

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        self.build_encode()


    def _loss(self):
        # Weight decay loss
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        mse_loss, pred, lab = MSE_Loss(self.outputs[0], 
                              self.labels, 
                              self.gen_labels)

        self.loss = self.loss + mse_loss
        self.pred = pred
        self.lab = lab
        self.mse_loss = mse_loss
        
        self.binary_reg_loss = binary_regularizer(self.outputs[0])

        self.loss += self.binary_reg_w * self.binary_reg_loss



class GraphHash_Emb_Code(GraphHash_Rank):
    def __init__(self, placeholders, input_dim, next_ele, **kwargs):
        super(GraphHash_Naive, self).__init__(**kwargs)

        self.graph_embs = []
        self.inputs = [next_ele[0], next_ele[1], next_ele[2]]
        self.labels = next_ele[3]
        self.gen_labels = None
        if FLAGS.label_type == 'ged':
            self.gen_labels = next_ele[4]
            
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.embedding_dim = FLAGS.embedding_dim
        self.output_dim = FLAGS.hash_code_len
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        self.build_encode()
            
           
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
       

        mlp_layers = []
        mlp_layers.append(Dense(input_dim=FLAGS.hidden1+FLAGS.hidden2+FLAGS.hidden3,
                                 output_dim=FLAGS.hidden4,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

        mlp_layers.append(Dense(input_dim=FLAGS.hidden4,
                                 output_dim=FLAGS.hidden5,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))


        mlp_layers.append(Dense(input_dim=FLAGS.hidden5,
                                 output_dim=self.embedding_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))

        
        discrete_layers = []
        discrete_layers.append(Dense(input_dim=self.embedding_dim,
                                 output_dim=FLAGS.hidden6,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))


        discrete_layers.append(Dense(input_dim=FLAGS.hidden6,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=FLAGS.dropout>0,
                                 logging=self.logging))
        
        self.layers = [conv_layers, pool_layers, mlp_layers, discrete_layers]


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

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
        
        for mlp_layer in self.layers[2]:
            hidden = mlp_layer(self.activations[-1])    
            self.activations.append(hidden)
            
        self.embeddings = self.activations[-1]


        for discrete_layer in self.layers[3]:
            hidden = discrete_layer(self.activations[-1])
            self.activations.append(hidden)
            
        self.outputs = self.activations[-1]
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def build_encode(self):
        placeholders = self.placeholders
        self.ecd_inputs = [placeholders['features'], 
                       placeholders['support'],
                       placeholders['graph_sizes']]
        try: 
            self.ecd_inputs[0] = tf.sparse.reorder(self.ecd_inputs[0])
            self.ecd_inputs[1] = tf.sparse.reorder(self.ecd_inputs[1])
        except AttributeError:
            self.ecd_inputs[0] = tf.sparse_reorder(self.ecd_inputs[0])
            self.ecd_inputs[1] = tf.sparse_reorder(self.ecd_inputs[1])
 
        self.ecd_activations = []
        self.ecd_activations.append(self.ecd_inputs)
        self.ecd_graph_embs = []
        # conv layers
        for conv_layer, pool_layer in zip(self.layers[0], self.layers[1]):
            hidden = conv_layer(self.ecd_activations[-1])
            graph_emb = pool_layer(hidden)
            self.ecd_graph_embs.append(graph_emb[0])
            self.ecd_activations.append(hidden)
        
        final_graph_emb = tf.concat(self.ecd_graph_embs, axis=1)
        self.ecd_activations.append([final_graph_emb, hidden[1], hidden[2]])
        
        for mlp_layer in self.layers[2]:
            hidden = mlp_layer(self.ecd_activations[-1])
            self.ecd_activations.append(hidden)
 
        self.ecd_embeddings = self.ecd_activations[-1]
      
        for discrete_layer in self.layers[3]:
            hidden = discrete_layer(self.ecd_activations[-1])
            self.ecd_activations.append(hidden)
        
        self.encode_outputs = self.ecd_activations[-1]
        self.codes = self.encode_outputs[0] > placeholders['thres']
        
    def _loss(self):
        # Weight decay loss
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        emb_mse_loss, _, _ = MSE_Loss(self.embeddings[0], 
                              self.labels, 
                              self.gen_labels)


        code_mse_loss, pred, lab = MSE_Loss(self.outputs[0], 
                                            self.labels, 
                                            self.gen_labels)

        self.loss = self.loss + code_mse_loss + emb_mse_loss
        self.pred = pred
        self.lab = lab
        self.code_mse_loss = code_mse_loss
        self.emb_mse_loss = emb_mse_loss

        self.loss += FLAGS.binary_regularizer_weight*\
                    binary_regularizer(self.outputs[0])
    

class GraphHash_Emb_Code_Binary(GraphHash_Emb_Code):
    def _loss(self):
        for layer_type in self.layers:
            for layer in layer_type:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        emb_mse_loss, _, _ = MSE_Loss(self.embeddings[0], 
                              self.labels, 
                              self.gen_labels)


        code_mse_loss, pred, lab = DSH_Loss(self.outputs[0], 
                                            self.labels, 
                                            self.gen_labels)

        self.loss = self.loss + code_mse_loss + emb_mse_loss
        self.pred = pred
        self.lab = lab
        self.code_mse_loss = code_mse_loss
        self.emb_mse_loss = emb_mse_loss

        self.loss += FLAGS.binary_regularizer_weight*\
                    binary_regularizer(self.outputs[0])