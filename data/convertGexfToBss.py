import networkx as nx
from glob import glob
import os
import pickle
import xml
has_dict = True
data_dir = "linux100/train"

graphs = []
geds = {}

constant =True
#def node_match(n1, n2):
#  return n1['type'] == n2['type']

output_file = open(data_dir+"/graphs.bss", 'w')

if has_dict == False:
  hashing = {}
  typeCnt = 0
else:
  dictionary = open('dict.pkl', 'rb')
  hashing = pickle.load(dictionary)
  typeCnt = len(hashing.keys())

count = 0
err_cnt = 0
for g_file in glob(data_dir+'/*.gexf'):
  gid = int(os.path.basename(g_file).split('.')[0])
  try:
    g = nx.read_gexf(g_file)
  except xml.etree.ElementTree.ParseError:
    continue

  g.graph['gid'] = gid

  label2node = {}

  output_file.write('%d\n'%gid)
  count = count + 1
  output_file.write("{} {}\n".format(len(g.nodes()), len(g.edges())))
  for i, n in enumerate(g.nodes(data=True)):
  #  print(n)  
    if constant:
      output_file.write('1\n')
    else:  
      if n[1]['atom'] not in hashing.keys():
        hashing[n[1]['atom']] = typeCnt
        typeCnt = typeCnt + 1
      output_file.write(str(hashing[n[1]['atom']])+'\n')
     
    #output_file.write('1\n')
    label2node[n[1]['label']] = i

  for e in g.edges():
    output_file.write("%d %d 1\n"%(label2node[e[0]],label2node[e[1]]))
  

output_file.close()
print('%d graphs in total'%count)

if has_dict == False:
  dictionary = open('dict.pkl', 'wb')
  pickle.dump(hashing, dictionary)

  dictionary.close()
