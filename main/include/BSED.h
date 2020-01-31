#ifndef _BSED_H
#define _BSED_H
#include "stdafx.h"
#include "treeNode.h"

typedef treeNode * PNode;
struct cmpPNode
{
	bool operator() (const PNode& lhs, const PNode& rhs) const
	{
		return  *rhs < *lhs;
	}
};

typedef priority_queue<PNode, vector<PNode >, cmpPNode > PQueue;
extern PNode succ[10000];
static int initFilter = 0;

typedef struct beamItem
{
	PNode left, right;
	int lower, upper;
	beamItem(int l, int u, PNode lf, PNode rg) : lower(l), upper(u), left(lf), right(rg){}
	beamItem()
	{
		left = right = NULL;
		lower = -1, upper = 0xffff;
	}
};

class BSEditDistance
{
public:
	vector<PQueue> closed;
	stack<beamItem>  bs;
	int w;
	int ged;
public:
	BSEditDistance(int width) { w = width; ged = 0xFFFF; }
	BSEditDistance(){ w = 50; ged = 0xFFFF; }
	~BSEditDistance()
	{

	}
private:
	inline void swap(PNode &a, PNode &b)
	{
		PNode c = a;
		a = b;
		b = c;
	}
	inline int median5(PNode *a)
	{
		PNode a0 = a[0];
		PNode a1 = a[1];
		PNode a2 = a[2];
		PNode a3 = a[3];
		PNode a4 = a[4];
		if (*a1 < *a0)
			swap(a0, a1);
		if (*a2 < *a0)
			swap(a0, a2);
		if (*a3 < *a0)
			swap(a0, a3);
		if (*a4 < *a0)
			swap(a0, a4);

		if (*a2 < *a1)
			swap(a1, a2);
		if (*a3 < *a1)
			swap(a1, a3);
		if (*a4 < *a1)
			swap(a1, a4);

		if (*a3 < *a2)
			swap(a2, a3);
		if (*a4 < *a2)
			swap(a2, a4);

		if (a2 == a[0]) return 0;
		if (a2 == a[1]) return 1;
		if (a2 == a[2]) return 2;
		if (a2 == a[3]) return 3;
		return 4;
	}
	inline int partition(PNode *a, int size, int pivot)
	{
		PNode pivotValue = a[pivot];
		swap(a[pivot], a[size - 1]);
		int storePos = 0;
		for (int loadPos = 0; loadPos < size - 1; loadPos++)
		{
			if (*a[loadPos] < *pivotValue)
			{
				swap(a[loadPos], a[storePos]);
				storePos++;
			}
		}
		swap(a[storePos], a[size - 1]);
		return storePos;
	}

	void freeNode(PNode &root)
	{
		if (root == 0) return;
		int size = root->childs.size();
		for (int i = 0; i < size; i++)
			freeNode(root->childs[i]);
		if (root)  delete root; root = 0;
	}

public:
	void select(PNode *a, int size, int k)
	{
		if (size < 5)
		{
			for (int i = 0; i<size; i++)
				for (int j = i + 1; j<size; j++)
					if (*a[j] < *a[i])
						swap(a[i], a[j]);
			return;
		}
		int groupNum = 0;
		PNode *group = &a[0];
		for (; groupNum * 5 <= size - 5; group += 5, groupNum++)
		{
			swap(group[median5(group)], a[groupNum]);
		}
		int numMedians = size / 5;
		int MOMIdx = numMedians / 2;
		select(a, numMedians, MOMIdx);
		int newMOMIdx = partition(a, size, MOMIdx);
		if (k != newMOMIdx)
		{
			if (k < newMOMIdx)
			{
				select(a, newMOMIdx, k);
			}
			else /* if (k > newMOMIdx) */
			{
				select(a + newMOMIdx + 1, size - newMOMIdx - 1, k - newMOMIdx - 1);
			}
		}
	}
	//here: we need to compute the w in O(n) time
	//first, determine the value of w
	int determine_wdith(PNode succ[], int &succ_size)
	{
		assert(succ_size != 0);

		int min_cost = succ[0]->ECost + succ[0]->deep;
		int count = 1, tmp;
		for (int i = 1; i < succ_size; i++)
		{
			int tmp = succ[i]->ECost + succ[i]->deep;
			if (tmp == min_cost){ count++; continue; }
			if (tmp < min_cost)
			{
				count = 1;
				min_cost = tmp;
			}
		}
		return count;
	}


	void pruneLayer(PNode succ[], PQueue &pq, int &succ_size)
	{
		if (!succ_size) return;
		int i = 0, size = succ_size;
		//int width = determine_wdith(succ, succ_size);
		//int width = 500000;
		//width = min(w, width);
		int width = w;
		if (size <= width)
		{
			for (; i < size; i++)
			{
				pq.push(succ[i]);
			}
			return;
		}
		select(&succ[0], size, width);
		for (; i < width; i++)
		{
			pq.push(succ[i]);
		}
		int minValue = succ[width]->deep + succ[width]->ECost;
		bs.top().upper = minValue;
		bs.top().right = succ[width]; //convert [0, u)->[L, u)
	}
	void expandSuccNode(vector<PNode > &node, PNode succ[], int &succ_size)
	{
		int size = node.size();
		for (int i = 0; i < size; i++)
		{
			if (!node[i]) continue;
			if (node[i]->visited == true)
			{
				freeNode(node[i]);
				continue;
			}
			int cost = node[i]->deep + node[i]->ECost;
			if (cost >= ged)
			{
				freeNode(node[i]);
				continue;
			}
			if (cost >= bs.top().lower && cost < bs.top().upper)
			{
				succ[succ_size++] = node[i];
			}
		}
	}
	void BeamSearchOnce(int &l, int &bound, vector<int> &group_1, vector<int> &group_2)
	{
		PQueue PQL = closed[l], PQLL, nullPQL;
		PNode node; int size;

		while (!PQL.empty() || !PQLL.empty())
		{
			size = 0;
			while (!PQL.empty())
			{
				node = PQL.top(); PQL.pop();
				if (node->allverifyGraphNodesUsed())
				{
					//assert(node->deep < bound);
					bound = node->deep;
					return;
				}
				if (!node->visited)
				{
					node->generateSuccessors(bound, group_1, group_2);
					node->visited = true;
					//totalExpandNode += node->childs.size();
					//total_search_node += node->childs.size();
				}
				expandSuccNode(node->childs, succ, size);
			}
			pruneLayer(succ, PQLL, size);
			PQL = PQLL; PQLL = nullPQL;
			if (PQL.empty() && PQLL.empty())
				return;
			l = l + 1;
			closed[l] = PQL;
			bs.push(beamItem(-1, bound, NULL, NULL));
		}
	}
	int getEditDistance(graph &ga, graph &gb, int bound, vector<int> &group_1, vector<int> &group_2)

	{
		int T = bound + 1, ged = bound + 1;
		int count = 0, l = 0;
		treeNode *start = new treeNode();
		start->init(ga, gb);
		int tmp_wdith = w;
		bool flag = false;

		PQueue startPQL;
		closed.resize(max(ga.v, gb.v) + 2, startPQL);
		bs.push(beamItem(-1, ged, NULL, NULL));

		startPQL.push(start);
		closed[0] = startPQL;

		while (!bs.empty())
		{
			//if (!flag) { w = 500; flag = !flag; }
			BeamSearchOnce(l, ged, group_1, group_2);
			while (!bs.empty() && bs.top().upper >= ged)
			{
				bs.pop();
				l--;
			}
			if (bs.empty())
			{
				if (start) freeNode(start);
				if (T == ged) return -1;
				else return ged;
			}
			bs.top().lower = bs.top().upper; bs.top().upper = ged; //
			bs.top().left = bs.top().right; bs.top().right = NULL;
			count++;
#if 0
			if (flag)
			{
				w = tmp_wdith;
			}
#endif
		}
		if (start) freeNode(start);
		if (T == ged) return -1;
		else return ged;
	}
public:
	inline int getEditDistance(graph &g1, graph &g2, int bound)
	{

		u8 lv_1[30000], lv_2[30000], le_1[64], le_2[64];
		memset(lv_1, 0, 30000 * sizeof(u8)); memset(lv_2, 0, 30000 * sizeof(u8));
		memset(le_1, 0, 64 * sizeof(u8)); memset(le_2, 0, 64 * sizeof(u8));
		max_v_1 = max_v_2 = max_e_1 = max_e_2 = 0;

		for (int i = 0; i < g1.v; i++)
		{
			for (int j = 0; j < g1.v; j++)
			{
				if (g1.E[i][j] != 0xff)
				{
					if (i < j)
					{
						le_1[g1.E[i][j]]++;
						if (max_e_1 < g1.E[i][j])
							max_e_1 = g1.E[i][j];
					}
				}
			}
			if (max_v_1 < g1.V[i]) max_v_1 = g1.V[i];
			lv_1[g1.V[i]]++;
		}
		max_v_1++; max_e_1++;

		for (int i = 0; i < g2.v; i++)
		{
			for (int j = 0; j < g2.v; j++)
			{
				if (g2.E[i][j] != 0xff)
				{
					if (i < j)
					{
						le_2[g2.E[i][j]]++;
						if (max_e_2 < g2.E[i][j])
							max_e_2 = g2.E[i][j];
					}
				}

			}
			if (max_v_2 < g2.V[i]) max_v_2 = g2.V[i];
			lv_2[g2.V[i]]++;
		}
		max_v_2++; max_e_2++;

		int commonVertex = common::initCommonLabel(lv_1, lv_2, max_v_1, max_v_2);
		int commonEdge = common::initCommonLabel(le_1, le_2, max_e_1, max_e_2);
		int lower_bound = max(g1.v, g2.v) - commonVertex + max(g1.e, g2.e) - commonEdge;
		if (lower_bound > bound) return -1;
/*
		int degree_1[1024], degree_2[1024];
		g1.degreeSet(degree_1, max_d_1); max_d_1++;
		g2.degreeSet(degree_2, max_d_2); max_d_2++;
		memset(tmpDegree1, 0, max_d_1 * sizeof(u8));
		memset(tmpDegree2, 0, max_d_2 * sizeof(u8));
		int i = 0, max1 = 0, max2 = 0, size1 = 0, size2 = 0, ie = 0, de = 0,
			edge1 = g1.e, edge2 = g2.e;

		for (int i = 0; i < g1.v; i++)
		{
			if (max1 < degree_1[i])
				max1 = degree_1[i];
			tmpDegree1[degree_1[i]]++;
		}
		for (i = max1; i >= 0; i--)
		{
			int len = tmpDegree1[i]; //chongdu
			for (int l = 0; l < len; l++)
			{
				if (size1 >= 512) cout << "size1 >= 512" << endl;
				ds1[size1++] = i;
			}
		}
		for (i = 0; i < g2.v; i++)
		{
			if (max2 < degree_2[i])
				max2 = degree_2[i];
			tmpDegree2[degree_2[i]]++;
		}
		for (i = max2; i >= 0; i--)
		{
			int len = tmpDegree2[i];
			for (int l = 0; l < len; l++)
			{
				if (size2 >= 512) cout << "size >= 512" << endl;
				ds2[size2++] = i;
			}
		}
		common::degreeEditDistance(ds1, size1, ds2, size2, ie, de);
		int tmp = max(2 * ie + edge1 - edge2, 2 * de + edge2 - edge1);
		tmp = max(tmp, edge2 - commonEdge + de);
		lower_bound = max(g1.v, g2.v) - commonVertex + tmp;
		if (lower_bound > bound) return -1;
*/
		if (false) return -1;
		else
		{
			initFilter++;
			u64 s1, s2;
			int group1_number, group2_number;
			if (FLAG)
			{
				group2.clear();
				s2 = g2.divideGroup(group2, group2_number); //here ?
				FLAG = false;
			}
			group1.clear(); s1 = g1.divideGroup(group1, group2_number);
			if (s1 > 1) VERTEXFLAG1 = true; else VERTEXFLAG1 = false;
			if (s2 > 1) VERTEXFLAG2 = true; else VERTEXFLAG2 = false;
			if (s1 < s2)
			{
				return getEditDistance(g1, g2, bound, group1, group2);
			}
			else
			{
				return getEditDistance(g2, g1, bound, group2, group1);
			}
		}
	}
};

#endif

