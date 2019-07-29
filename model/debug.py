import os
from config import FLAGS

print('Start testing')
#total_query_num = data_fetcher.get_test_graphs_num()
# Read Ground Truth
ground_truth = {}
ground_truth_path = os.path.join('..','data',
                                 FLAGS.dataset,
                                 'test',
                                 FLAGS.ground_truth_file)
f = open(ground_truth_path, 'r')
ged_cnt = {}
for line in f.readlines():
    g, q, d = line.split(' ')
    g = int(g)
    q = int(q)
    d = int(d)
    if q not in ground_truth.keys():
        ground_truth[q] = []
    ground_truth[q].append((g,d))
    ged_cnt.setdefault(d,0)
    ged_cnt[d] = ged_cnt[d] + 1

PAtKs = []
SRCCs = []
KRCCs = []
t_max = FLAGS.GED_threshold - 2
precisions = [[] for i in range(t_max)]
recalls = [[] for i in range(t_max)]
f1_scores = [[] for i in range(t_max)]
zero_cnt = [0 for i in range(t_max)]

total_query_num = 100
encode_batchsize = 15
#
d = None
for q in ground_truth.keys():
    if d is None:
        d = q
    ground_truth[q] = sorted(ground_truth[q], key=lambda x: x[1]*10000000 + x[0])
    if ground_truth[q][0][1] < 7:
        print(ground_truth[q][0])

print(ground_truth[d])
