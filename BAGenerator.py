# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:41:02 2019

@author: Zongyue
"""


import networkx as nx
import scipy.stats as stats
import numpy as np

def write_graph(graph, f, n):
    f.write('%d\n'%n)
    for i, node in graph.nodes(data=True):
        f.write(str(node[1]['type'])+'\n')
    for e in g.edges():
        f.write("%d %d 1\n"%(node[1]['label'],node[1]['label']))



train_file = "BA/train/graphs.bss"
test_file = "BA/test/graphs.bss"

N = 49999900
cnt = 0
ave_v_num = 10

p_low = 0.3
p_high = 1

alpha = np.array([0.8, 0.1, 0.05, 0.025, 0.025])
theta = np.zeros(5)
s = 0
for i in range(alpha.size):
    s = s + alpha[i]
    theta[i] = s
    

f = open(train_file, 'w')
while cnt < N:
    if cnt % 1000000 == 0:
        print('finish %d'%cnt)
        
    n = np.random.poisson(lam=ave_v_num)
    while n > 15:
        n = np.random.poisson(lam=ave_v_num)
        
    m = np.random.randint(n)
    G = nx.generators.random_graphs.barabasi_albert_graph(n,m)
    newG = nx.Graph()
    for node in G.nodes():
        t = np.random.uniform()
        for i in range(alpha.size-1):
            if t < theta[i+1]:
                newG.add_node(node, type=i)
                f = True
                break
    if len(G.nodes()) != len(newG.nodes()):
        print('error')
    for edge in G.edges():
        newG.add_edge(*edge)
    write_graph(newG, f, cnt)
    cnt = cnt + 1
f.close()

f = open(test_file, 'w')
while cnt < 100:
    n = np.random.poisson(lam=ave_v_num)
    while n > 15:
        n = np.random.poisson(lam=ave_v_num)
        
    m = np.random.randint(n)
    G = nx.generators.random_graphs.barabasi_albert_graph(n,m)
    newG = nx.Graph()
    for node in G.nodes():
        t = np.random.uniform()
        for i in range(alpha.size-1):
            if t < theta[i+1]:
                newG.add_node(node, type=i)
                f = True
                break
    if len(G.nodes()) != len(newG.nodes()):
        print('error')
    for edge in G.edges():
        newG.add_edge(*edge)
    write_graph(newG, f, cnt)
    cnt = cnt + 1
f.close()

