#ifndef  _TREE_Node_H
#define  _TREE_Node_H
#include "verifyGraph.h"
#include "global.h"

class treeNode
{
public:
	verifyGraph uG1, uG2;	
	int deep, ECost; //real cost and estimate cost
	int CVLabel, CELabel;	
	int *group, *cost;
	u8 *matching, *inverseMatching;
	u8 *degree1, *degree2;
	u8 *lv1, *lv2;
	u8 *le1, *le2;
	bool visited; // 
	vector<treeNode *> childs;

public:
	treeNode()
	{
		this->ECost = this->deep = 0;
		this->matching = this->inverseMatching = 0;
		this->CVLabel = this->CELabel = 0;
		this->visited = false;

		lv1 = lv2 = 0;
		le1 = le2 = 0;
		degree1 = degree2 = 0;
		group = cost = 0;

	}
	treeNode(const treeNode &tn) //deep copy
	{
		*this = tn;
	}
	treeNode & operator= (const treeNode &tn) //deep copy
	{
		if (this != &tn)
		{			
			this->uG1 = tn.uG1; this->uG2 = tn.uG2;
			matching = new u8[uG1.gs]; memcpy(matching, tn.matching, sizeof(u8) * uG1.gs);
			inverseMatching = new u8[uG2.gs];
			memcpy(inverseMatching, tn.inverseMatching, sizeof(u8) * uG2.gs);

			this->lv1 = new u8[max_v_1]; memcpy(lv1, tn.lv1, max_v_1);
			this->lv2 = new u8[max_v_2]; memcpy(lv2, tn.lv2, max_v_2);
			this->le1 = new u8[max_e_1]; memcpy(le1, tn.le1, max_e_1);
			this->le2 = new u8[max_e_2]; memcpy(le2, tn.le2, max_e_2);
			this->cost = new int[uG1.gs]; memcpy(this->cost, tn.cost, sizeof(int) * uG1.gs);

			this->degree1 = new u8[uG1.gs]; 
			this->degree2 = new u8[uG2.gs];
			#if 1 //here: ? 
			if (VERTEXFLAG1)
			{
				this->group = new int[uG1.gs];
				memcpy(this->group, tn.group, sizeof(int) * uG1.gs);
			}
			else
				this->group = 0;
			#endif
			this->visited = false;
		}
		return *this;
	}
	friend bool operator< (const treeNode &aa, const treeNode &bb)// const 
	{

		int costa = aa.deep + aa.ECost;
		int costb = bb.deep + bb.ECost;
		if (costa < costb)
			return true;
		else if (costa > costb)
			return false;
		else
		{
			if (aa.deep < bb.deep)
				return true;
			else
				return false;
		}
	}

	~treeNode()
	{
		if (matching)
		{
			delete[]matching; matching = 0;
		}
		if (inverseMatching)
		{
			delete[] inverseMatching; inverseMatching = 0;
		}
		if (lv1)
		{
			delete[] lv1; lv1 = 0;
		}
		if (lv2)
		{
			delete[] lv2; lv2 = 0;
		}
		if (le1)
		{
			delete[] le1; le1 = 0;
		}
		if (le2)
		{
			delete[] le2; le2 = 0;
		}
		if (degree1)
		{
			delete[] degree1; degree1 = 0;
		}
		if (degree2)
		{
			delete[] degree2; degree2 = 0;
		}
#if 1
		if (group)
		{
			delete[] group; group = 0;
		}
#endif
		if (cost)
		{
			delete[] cost; cost = 0;
		}
		if (childs.size() > 0)
		{
			vector<treeNode *>().swap(childs);
		}
	}
public:	
	bool allverifyGraphNodesUsed()
	{
		if (uG1.v == 0 && uG2.v == 0)
			return true;
		return false;
	}
	inline int getNumberOfAdjacentverifyGraphEdges(u8 * &m, vector<vector<int > > &adjList, int i)
	{
		int e = 0;
		for (int j = 0; j < adjList[i].size(); j++)
		{
			int idx = adjList[i][j];
			if (m[idx] != UNMAPPED) e += 1;
		}
		return e;
	}
	int processverifyGraphEdges(int &startIndex, int &endIndex)
	{
		int tmp_cost = 0;
		for (int e = 0; e < startIndex; e++)
		{
			int end2Index = this->matching[e];
			if (a1[startIndex][e] != 0xff)
			{
				if (end2Index < uG2.gs && a2[endIndex][end2Index] != 0xff)
				{
					int verifyGraphEdge = a1[startIndex][e];
					int verifyGraphEdge2 = a2[endIndex][end2Index];
					if (verifyGraphEdge != verifyGraphEdge2) //subtitution 
						tmp_cost++;
					else
						; //no edit operation
				}
				else //deletion 
					tmp_cost++;							
			}
			else
			{
				if (end2Index < uG2.gs && a2[endIndex][end2Index] != 0xff) //insertion	
					tmp_cost++;								
			}
		}
		return tmp_cost;
	}
	inline void updateVertexDegree(vector<vector<int > > &adjList, u8 *degree, int &pos)
	{
		if (pos == DELETED) return;
		for (int i = 0; i < adjList[pos].size(); i++)
		{
			int vertexAdj = adjList[pos][i];
			degree[vertexAdj]--;
		}
	}
	inline void updateCommonLabel(int &x, int f1, int &y, int f2, int &cl)
	{
		assert(x >= 0 && y >= 0);
		if (x < y && x + f2 < y + f1) cl = cl - f1;
		else if (x < y && x + f2 >= y + f1) cl = cl + y - (x + f2);
		else if (x >= y && x + f2 < y + f1) cl = cl + x - (y + f1);
		else
			cl = cl - f2;
	}
public:
	void init(graph &g1, graph &g2);	
	int  computeCVL(u8 *lv1, int li, u8 *lv2, int lj);
	void updateCVL(u8 *lv1, int li, u8 *lv2, int lj, int &cvl);
	int  computeCEL(int &start, int &end, int &e1, int &e2);
	void updateCEL(int &start, int &end, int &e1, int &e2, int &cel);
	int  degree_distance_1(int &le1, int &le2, int &v1, int &v2, int &startIndex, int &endIndex,
		vector<int> &vs, int ranki);
	int edgeOutDistance(int &startIndex, int &endIndex, int &v1, int &v2,
		int &EI, int &ED, int &ES, int &A, int &NA, int &NB);
	int labelEditDistance(int &startIndex, int &endIndex, int &cost, int &v1, int &v2, int &cvl,
		int &e1, int &e2, int &cel, int &bound, bool &flag);
	void generateSuccessors(int &bound, vector<int> &group_1, vector<int> &group_2);	
};
#endif
