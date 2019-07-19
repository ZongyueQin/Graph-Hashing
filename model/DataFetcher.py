# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:20:31 2019

@author: Zongyue Qin
"""
import os
from random import shuffle, sample
from glob import glob
import networkx as nx
import scipy.sparse as sp
import sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np


from config import FLAGS
from utils import sorted_nicely

class DataFetcher:
    """ Represents a set of data """
    
    """ read training graphs and test graphs """
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_type = FLAGS.node_feat_encoder
        self.node_feat_name = FLAGS.node_feat_name
        
        self.data_dir = os.path.join('..','data',dataset)
        self.train_data_dir = os.path.join(self.data_dir,'train')
        self.test_data_dir = os.path.join(self.data_dir, 'test')
        
        train_graphs = self._readGraphs(self.train_data_dir)
        test_graphs = self._readGraphs(self.test_data_dir)
        
        train_graphs, valid_graphs = self._split_train_valid(train_graphs)
        
        self.node_feat_encoder = self._create_node_feature_encoder(
            train_graphs + valid_graphs + test_graphs)

        self.train_graphs = self.create_MyGraph(train_graphs, 'train')
        self.valid_graphs = self.create_MyGraph(valid_graphs, 'valid')
        self.test_graphs = self.create_MyGraph(test_graphs, 'test')

    def get_train_graph_gid(self, pos):
        return self.train_graphs[pos].nxgraph.graph['gid']

    def get_node_feature_dim(self):
        if len(self.train_graphs) == 0:
            raise RuntimeError('train_graphs is empty, can\'t get feature dim')
        return self.train_graphs[0].sparse_node_inputs.shape[1]

    def get_data_train_and_merge(self, batchsize, max_mat_size = -1):
        sample_graphs = sample(self.train_graphs, 2 * batchsize)
        
        features = sp.vstack([g.sparse_node_inputs for g in sample_graphs])
        features = self._sparse_to_tuple(features)
        
        laplacians = sp.block_diag([g.laplacian for g in sample_graphs])
        laplacians = self._sparse_to_tuple(laplacians)
        
        sizes = [g.nxgraph.number_of_nodes() for g in sample_graphs]
        
        labels = self._get_labels(sample_graphs[0:batchsize],
                                  sample_graphs[batchsize:])

        return features, laplacians, sizes, labels

    def get_data_from_train_to_encode(self, idx_list = None):
        graphs = []
        if idx_list is None:
            graphs = self.train_graphs
        else:
            for idx in idx_list:
                graphs.append(self.train_graphs[idx])
        
        features = sp.vstack([g.sparse_node_inputs for g in graphs])
        features = self._sparse_to_tuple(features)
        
        laplacians = sp.block_diag([g.laplacian for g in graphs])
        laplacians = self._sparse_to_tuple(laplacians)
        
        sizes = [g.nxgraph.number_of_nodes() for g in graphs]
                
        return features, laplacians, sizes


    def get_data_from_test_to_encode(self, idx_list = None):
        graphs = []
        if idx_list is None:
            graphs = self.test_graphs
        else:
            for idx in idx_list:
                graphs.append(self.test_graphs[idx])
        
        features = sp.vstack([g.sparse_node_inputs for g in graphs])
        features = self._sparse_to_tuple(features)
        
        laplacians = sp.block_diag([g.laplacian for g in graphs])
        laplacians = self._sparse_to_tuple(laplacians)
        
        sizes = [g.nxgraph.number_of_nodes() for g in graphs]
                
        return features, laplacians, sizes


    def get_train_graphs_num(self):
        return len(self.train_graphs)
    
    def get_test_graphs_num(self):
        return len(self.test_graphs)
    """
    def get_data_train_in_pair(self, batchsize, max_mat_size = -1):
        if max_mat_size == -1:
            max_mat_size = sys.maxsize
        
        # Randomly sample n pairs of graphs
        # TODO this part needs thorough consideration for good 
        # sampling strategy
        sample_graphs = sample(self.train_graphs, 2 * batchsize)
        graphs_1 = sample_graphs[0:batchsize]
        graphs_2 = sample_graphs[batchsize:]
        # TODO Compute if they are similar
        labels = self._get_labels(graphs_1, graphs_2)
        
        # stitch their laplacian matrices and features together
        features_1 = sp.vstack([g.sparse_node_inputs for g in graphs_1])
        features_1 = self._sparse_to_tuple(features_1)
        
        features_2 = sp.vstack([g.sparse_node_inputs for g in graphs_2])
        features_2 = self._sparse_to_tuple(features_2)
        
        laplacians_1 = sp.block_diag([g.laplacian for g in graphs_1])
        laplacians_1 = self._sparse_to_tuple(laplacians_1)
        
        laplacians_2 = sp.block_diag([g.laplacian for g in graphs_2])
        laplacians_2 = self._sparse_to_tuple(laplacians_2)
        
        sizes_1 = [g.nxgraph.number_of_nodes() for g in graphs_1]
        sizes_2 = [g.nxgraph.number_of_nodes() for g in graphs_2]
        
        # return laplacian matrix and feature matrix
        return features_1, laplacians_1, sizes_1, features_2, laplacians_2, sizes_2, labels
    """        
    """ read *.gexf in graph_dir and return a list of networkx graph """
    def _readGraphs(self, graph_dir):
        graphs = []
        for file in sorted_nicely(glob(graph_dir+'/*.gexf')):
            gid = int(os.path.basename(file).split('.')[0])
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            graphs.append(g)
            if not nx.is_connected(g):
                print('{} not connected'.format(gid))
                
        return graphs
    
    """ split train_graphs into training set and validation set """
    def _split_train_valid(self, graphs):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0,1]'.format(
                    self.valid_percentage))
        shuffle(graphs)
        sp = int(len(graphs) * self.valid_percentage)
        valid_graphs = graphs[0:sp]
        train_graphs = graphs[sp:]
        
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    """ make sure the number of graphs > 2 """
    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(
                label, len(graphs)))
    
    def _create_node_feature_encoder(self, gs):
        if self.node_feat_type == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        elif 'constant' in self.node_feat_type:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

    def create_MyGraph(self, graphs, tvt):
        rtn = []
        hits = [0, 0.3, 0.6, 0.9]
        cur_hit = 0
        for i, g in enumerate(graphs):
            mg = MyGraph(g, self.node_feat_encoder)
            perc = i / len(graphs)
            if cur_hit < len(hits) and abs(perc - hits[cur_hit]) <= 0.05:
                print('{} {}/{}={:.1%}'.format(tvt, i, len(graphs), i / len(graphs)))
                cur_hit += 1
            rtn.append(mg)
        return rtn

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def _get_labels(self, graphs_1, graphs_2):
        labels = [0 for g in graphs_1]
        for i, g_pair in enumerate(zip(graphs_1, graphs_2)):
            #ged = graph_edit_distance(g_pair[0].nxgraph, g_pair[1].nxgraph, 
            #                             upper_bound=FLAGS.GED_threshold,
            #                             node_match=node_match)
            ged = abs(len(g_pair[0].nxgraph.nodes())-len(g_pair[0].nxgraph.nodes()))
            if ged <= FLAGS.GED_threshold:
                labels[i] = 1
        return labels
            

"""------------------------------------------------------------------------"""
        
     
""" My wrapper object for networkx graph 
    Parts of the code refer to https://github.com/tkipf/gcn """
class MyGraph(object):
    
    def __init__(self, nxgraph, node_feat_encoder):
        self.nxgraph = nxgraph
        adj_mat = nx.to_scipy_sparse_matrix(nxgraph)
        dense_node_inputs = node_feat_encoder.encode(nxgraph)
        # TODO probably add ordering
        
        self.sparse_node_inputs = self._preprocess_inputs(
                sp.csr_matrix(dense_node_inputs))
        
        self.laplacian = self._preprocess_adj(adj_mat)
    
        
    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return inputs
#        return self._sparse_to_tuple(inputs)

    
    """ compute laplacian matrix, return a sparse matrix in form of tuple """
    def _preprocess_adj(self, adj_mat):
        adj_proc = sp.coo_matrix(adj_mat+sp.eye(adj_mat.shape[0]))
        if FLAGS.laplacian == 'gcn': # what's other options ?
            adj_proc = self._normalize_adj(adj_proc)
        
        return adj_proc
#        return self._sparse_to_tuple(adj_proc)
    
    """ Symmetrically normalize matrix """
    def _normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class NodeFeatureEncoder(object):
    def encode(self, g):
        raise NotImplementedError()

    def input_dim(self):
        raise NotImplementedError()


class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder().fit(
            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))

    def add_new_feature(self, feat_name):
        """Use this function if a new feature was added to the graph set."""
        # Add the new feature to the dictionary
        # as a unique feature and reinit the encoder.
        new_idx = len(self.feat_idx_dic)
        self.feat_idx_dic[feat_name] = new_idx
        self._fit_onehotencoder()

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    def __init__(self, _, node_feat_name):
        self.input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)
        return rtn

    def input_dim(self):
        return self.input_dim_
