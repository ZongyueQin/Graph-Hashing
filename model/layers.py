from config import FLAGS
import tensorflow as tf
from inits import glorot, zeros

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def mul_3D2D(tensor, mat, interact_dim):
    ten_shape = tf.shape(tensor)
    mat_shape = tf.shape(mat)
    ret_shape = tf.concat([ten_shape[0:2], mat_shape[1:]], axis=0)

    mid_res = tf.reshape(tensor, [-1, interact_dim])
    mid_res = tf.matmul(mid_res, mat)
    return tf.reshape(mid_res, ret_shape)

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] = _LAYER_UIDS[layer_name] + 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor = random_tensor+tf.random_uniform(shape=tf.reshape(noise_shape,[1]))
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def mat_vec_dot(x, y):
    """ implement of matrix-vector multiplication, x is a matrix, y is vector"""
    return tf.reduce_sum(tf.multiply(x, y), axis=1)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        """ The input always have three elements: feature, laplacian and sizes
            of every graph. And return also contains these three parts. """
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs[0])
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs[0])
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        #self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[0]
        # dropout
        #if self.sparse_inputs:
        #    x = sparse_dropout(x, 1-self.dropout, tf.reduce_sum(inputs[2]))
        #else:
        #    x = tf.nn.dropout(x, rate=self.dropout)
        

        # transform
        with tf.variable_scope(self.name + '_vars'):
            # dropout
            if self.dropout:
                weights = tf.nn.dropout(self.vars['weights'], 
                                    1 - self.dropout)
                if self.bias:
                    bias = tf.nn.dropout(self.vars['bias'], 
                                     1 - self.dropout)
            else:
                weights = self.vars['weights']
                if self.bias:
                    bias = self.vars['bias']


            output = dot(x, weights, sparse=self.sparse_inputs)

            # bias
            if self.bias:
                output = output + bias

        return [self.act(output), inputs[1], inputs[2]]


class GraphConvolution_GCN(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_GCN, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        #self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope(self.name + '_vars'):
            
            self.vars['weights'] = glorot([input_dim, output_dim],
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[0]

        # dropout
#        if self.sparse_inputs:
#            x = sparse_dropout(x, 1-self.dropout, tf.reduce_sum(inputs[2]))
#        else:
#            x = tf.nn.dropout(x, rate=self.dropout)

        # convolve
        with tf.variable_scope(self.name + '_vars'):
            if self.dropout:
            # dropout
                weights = tf.nn.dropout(self.vars['weights'], 
                                    1 - self.dropout)
                if self.bias:
                    bias = tf.nn.dropout(self.vars['bias'], 
                                     1 - self.dropout)

            else:
                weights = self.vars['weights']
                if self.bias:
                    bias = self.vars['bias']

            if not self.featureless:
                pre_sup = dot(x, 
                              weights,
                              sparse=self.sparse_inputs)
            else:
                pre_sup = weights
        
        
            output = dot(inputs[1], pre_sup, sparse=True)
        
            # bias
            if self.bias:
                output = output + bias

        return [self.act(output),inputs[1], inputs[2]]
    
class SplitAndMeanPooling(Layer):
    def __init__(self,**kwargs):
        super(SplitAndMeanPooling, self).__init__(**kwargs)
    
    def _call(self, inputs):
        features_list = tf.split(inputs[0], inputs[2])
        graph_emb_list = []
        for features in features_list:
            # Generate a graph embedding per graphs
            graph_emb_list.append(tf.reduce_mean(features, axis=0))
        output = tf.stack(graph_emb_list,axis=0)
        return [output, inputs[1], inputs[2]]

class SplitAndAttentionPooling(Layer):
    def __init__(self, input_dim, 
                 placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False,
                 **kwargs):
        super(SplitAndAttentionPooling, self).__init__(**kwargs)
        
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.bias = bias
        # the output dimension is same as input dimension
        self.output_dim = input_dim
        # helper variable for sparse dropout
        #self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope(self.name + '_vars'):
            
            self.vars['weights'] = glorot([input_dim, input_dim],
                                                        name='weights')
            if self.bias:
                self.vars['bias'] = zeros([input_dim], name='bias')

        if self.logging:
            self._log_vars()
            
    def _call(self, inputs):
        features_list = tf.split(inputs[0], inputs[2])
        graph_emb_list = []
        
        with tf.variable_scope(self.name + '_vars'):
            if self.dropout:
            # dropout
                weights = tf.nn.dropout(self.vars['weights'], 
                                    1 - self.dropout)
                if self.bias:
                    bias = tf.nn.dropout(self.vars['bias'], 
                                     1 - self.dropout)
            else:
                weights = self.vars['weights']
                if self.bias:
                    bias = self.vars['bias']


            for features in features_list:
                # Generate a graph embedding per graphs
                graph_mean = tf.reduce_mean(features, axis=0)
                global_feat = mat_vec_dot(weights, graph_mean)
                
                if self.bias:
                    global_feat = global_feat + bias
                
                global_feat = tf.nn.relu(global_feat)

                attention = self.act(mat_vec_dot(features, global_feat))
                attention = tf.squeeze(attention)

                graph_emb = tf.transpose(tf.multiply(attention, 
                                                     tf.transpose(features)))
                graph_emb = tf.reduce_sum(graph_emb, axis=0)
                graph_emb_list.append(graph_emb)
            
            output = tf.stack(graph_emb_list,axis=0)

        return [output, inputs[1], inputs[2]]

class SplitIntoPairs(Layer):
    def __init__(self, **kwargs):
        super(SplitIntoPairs, self).__init__(**kwargs)
            
    def _call(self, inputs):
        x = inputs[0]
        bs = FLAGS.batchsize
        k = FLAGS.k
        
        x1, x2 = tf.split(x, [bs, bs*k])
        x2 = tf.reshape(x2, [bs, k, -1])
        tmp = tf.expand_dims(x1, 1)
        tmp = tf.tile(tmp, [1, bs, 1])
       # tmp = tf.stack([x1 for i in range(bs)], axis=1)
        x2 = tf.concat([tmp, x2], axis=1)
        
        x = [x1, x2]
        return [x, inputs[1], inputs[2]]
    
class NTN(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 act=tf.nn.relu, bias=False,
                 **kwargs):
        super(NTN, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        #self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.variable_scope(self.name + '_vars'):
            
            self.vars['W'] = glorot([output_dim, input_dim, input_dim],
                                                        name='W')
            self.vars['V'] = glorot([output_dim, 2*input_dim, 1],
                                     name='V')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x1 = inputs[0][0]
        x2 = inputs[0][1]
        x2_t = tf.transpose(x2, perm=[0,2,1])
        # convolve
        with tf.variable_scope(self.name + '_vars'):
            if self.dropout:
            # dropout
                W = tf.nn.dropout(self.vars['W'], 
                                    1 - self.dropout)
                V = tf.nn.dropout(self.vars['V'],
                                  1 - self.dropout)
                if self.bias:
                    bias = tf.nn.dropout(self.vars['bias'], 
                                     1 - self.dropout)

            else:
                W = self.vars['W']
                V = self.vars['V']
                if self.bias:
                    bias = self.vars['bias']

            output_list = []
            for i in range(self.output_dim):
                mid_res_1 = tf.matmul(x1, W[i])
#                mid_res_1 = tf.reshape(mid_res_1, [bs, 1, self.input_dim])
                mid_res_1 = tf.expand_dims(mid_res_1, 1)
                mid_res_1 = tf.matmul(mid_res_1, x2_t)
                mid_res_1 = tf.squeeze(mid_res_1)
                
#                mid_res_2 = tf.reshape(x1, [bs, 1, self.input_dim])
                mid_res_2 = tf.expand_dims(x1, 1)
                multiples = tf.cast(tf.shape(x2)/tf.shape(mid_res_2), tf.int32)
                mid_res_2 = tf.tile(mid_res_2, multiples)
                mid_res_2 = tf.concat([mid_res_2, x2], axis=2)
                mid_res_2 = mul_3D2D(mid_res_2, V[i], 
                                     2*self.input_dim)
                mid_res_2 = tf.squeeze(mid_res_2)
                
                mid_res = mid_res_1+mid_res_2
                if self.bias:
                    mid_res = mid_res + bias[i]
                mid_res = tf.reshape(mid_res, [-1,1])
                output_list.append(mid_res)
                
            output = tf.concat(output_list, axis=1)


        return [self.act(output),inputs[1], inputs[2]]

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs0', inputs[0][0])
                tf.summary.histogram(self.name + '/inputs1', inputs[0][1])
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs[0])
            return outputs

 
