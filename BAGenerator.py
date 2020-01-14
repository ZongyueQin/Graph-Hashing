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
    f.write('{} {}\n'.format(len(graph.nodes()), len(graph.edges())))
    for i, node in graph.nodes(data=True):
        f.write(str(node['type'])+'\n')
    for e in graph.edges():
        f.write("%d %d 1\n"%(e[0],e[1]))



train_file = "BA/train/graphs.bss"
test_file = "BA/test/graphs.bss"

N = 499#99900
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
print(theta)
    

file = open(train_file, 'w')
while cnt < N:
    if cnt % 1000000 == 0:
        print('finish %d'%cnt)
        
    n = np.random.poisson(lam=ave_v_num)
    while n > 15 or n < 7:
        n = np.random.poisson(lam=ave_v_num)
        
    m = np.random.randint(1, n)
    G = nx.generators.random_graphs.barabasi_albert_graph(n,m)
    newG = nx.Graph()
    for node in G.nodes():
        t = np.random.uniform()
        for i in range(alpha.size):
            if t < theta[i]:
                newG.add_node(node, type=i)
                f = True
                break
    if len(G.nodes()) != len(newG.nodes()):
        print('error')
    for edge in G.edges():
        newG.add_edge(*edge)
    write_graph(newG, file, cnt)
    cnt = cnt + 1
file.close()

file = open(test_file, 'w')
while cnt < N+100:
    n = np.random.poisson(lam=ave_v_num)
    while n > 15 and n < 7:
        n = np.random.poisson(lam=ave_v_num)
        
    m = np.random.randint(1, n)
    G = nx.generators.random_graphs.barabasi_albert_graph(n,m)
    newG = nx.Graph()
    for node in G.nodes():
        t = np.random.uniform()
        for i in range(alpha.size):
            if t < theta[i]:
                newG.add_node(node, type=i)
                f = True
                break
    if len(G.nodes()) != len(newG.nodes()):
        print('error')
    for edge in G.edges():
        newG.add_edge(*edge)
    write_graph(newG, file, cnt)
    cnt = cnt + 1
file.close()

