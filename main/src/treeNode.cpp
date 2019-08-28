#include "stdafx.h"
#include "treeNode.h"

void treeNode::init(graph &g1, graph &g2)
{
	uG1.init(g1); degree1 = new u8[uG1.gs]; g1.degreeSet(degree1, max_d_1); max_d_1++;
	uG2.init(g2); degree2 = new u8[uG2.gs]; g2.degreeSet(degree2, max_d_2); max_d_2++;
	memset(a1, 0xff, 512 * 512); memset(a2, 0xff, 512 * 512);
	adjList1.clear(); adjList2.clear();
	max_v_1 = max_v_2 = max_e_1 = max_e_2 = 0;

	if (VERTEXFLAG1)
	{
		this->group = new int[uG1.gs];
		for (int i = 0; i < uG1.gs; i++)
			group[i] = -1;
	}
	for (int i = 0; i < g1.v; i++)
	{
		vector<int > tmp;
		gn1[i].verifyGraphNodeID = i;
		gn1[i].verifyGraphNodeStr = g1.V[i];
		for (int j = 0; j < g1.v; j++)
		{
			a1[i][j] = g1.E[i][j];
			if (a1[i][j] != 0xff)
			{
				tmp.push_back(j);
				if (i < j)
				{
					if (max_e_1 < a1[i][j])
						max_e_1 = a1[i][j];
				}

			}
		}
		if (max_v_1 < g1.V[i]) max_v_1 = g1.V[i];
		adjList1.push_back(tmp);
	}
	max_v_1++; max_e_1++;

	for (int i = 0; i < g2.v; i++)
	{
		vector<int > tmp;
		gn2[i].verifyGraphNodeID = i;
		gn2[i].verifyGraphNodeStr = g2.V[i];
		for (int j = 0; j < g2.v; j++)
		{
			a2[i][j] = g2.E[i][j];
			if (a2[i][j] != 0xff)
			{
				tmp.push_back(j);
				if (i < j)
				{
					if (max_e_2 < a2[i][j])
						max_e_2 = a2[i][j];
				}
			}

		}
		if (max_v_2 < g2.V[i]) max_v_2 = g2.V[i];
		adjList2.push_back(tmp);
	}
	max_v_2++; max_e_2++;
	lv1 = new u8[max_v_1]; memset(lv1, 0, max_v_1);
	lv2 = new u8[max_v_2]; memset(lv2, 0, max_v_2);
	le1 = new u8[max_e_1]; memset(le1, 0, max_e_1);
	le2 = new u8[max_e_2]; memset(le2, 0, max_e_2);
	int size, to;

	for (int i = 0; i < uG1.gs; i++)
	{
		size = adjList1[i].size();
		lv1[g1.V[i]]++;
		for (int j = 0; j < size; j++)
		{
			to = adjList1[i][j];
			if (i < to)
				le1[a1[i][to]]++;
		}
	}
	for (int i = 0; i < uG2.gs; i++)
	{
		lv2[g2.V[i]]++;
		size = adjList2[i].size();
		for (int j = 0; j < size; j++)
		{
			to = adjList2[i][j];
			if (i < to)
				le2[a2[i][to]]++;
		}
	}

	this->matching = new u8[uG1.gs];
	memset(matching, 0xff, uG1.gs * sizeof(u8));
	this->inverseMatching = new u8[uG2.gs];
	memset(inverseMatching, 0xff, uG2.gs * sizeof(u8));
	this->cost = new int[uG1.gs];
	memset(cost, 0, uG1.gs * sizeof(int));

	this->deep = 0;
	this->CVLabel = common::initCommonLabel(this->lv1, this->lv2, max_v_1, max_v_2);
	this->CELabel = common::initCommonLabel(this->le1, this->le2, max_e_1, max_e_2);
	this->ECost = max(g1.v, g2.v) - this->CVLabel + max(g1.e, g2.e) - this->CELabel;
	this->visited = false;
	this->cost[0] = this->deep + this->ECost;

}
int treeNode::computeCVL(u8 *lv1, int li, u8 *lv2, int lj)
{
	int before_CVL = this->CVLabel;
	int fj1 = lv1[li], fj2 = li < max_v_2 ? lv2[li] : 0;
	assert(fj1 >= 0 && fj2 >= 0);
	if (lj == DELETED)
	{
		if (fj1 <= fj2) before_CVL--;
		return before_CVL;
	}
	else
	{
		if (lj == li) before_CVL--;
		else
		{
			int fi1 = lj < max_v_1 ? lv1[lj] : 0, fi2 = lv2[lj];
			assert(fi1 >= 0 && fi2 >= 0);
			if (fj1 > fj2 && fi1 < fi2);
			else if (fj1 > fj2 && fi1 >= fi2)
			{
				before_CVL--;
			}
			else if (fj1 <= fj2 && fi1 < fi2)
			{
				before_CVL--;
			}
			else
			{
				before_CVL = before_CVL - 2;
			}
		}
	}
	return before_CVL;
}
void treeNode::updateCVL(u8 *lv1, int li, u8 *lv2, int lj, int &cvl)
{
	this->CVLabel = cvl;
	if (lj == DELETED)
	{
		lv1[li]--; 
		return;
	}
	lv1[li]--;
	lv2[lj]--;
	
}
int treeNode::computeCEL(int &start, int &end, int &e1, int &e2)
{
	int before_CEL = this->CELabel;
	e1 = this->uG1.e, e2 = this->uG2.e;

	memset(start_deleted, 0, sizeof(u8) * max_e_1);
	int max_1 = -1, max_2 = -1;

	for (int i = 0; i < adjList1[start].size(); i++)
	{
		int to = adjList1[start][i];
		if (!this->uG1.flag[to])
		{
			start_deleted[a1[start][to]]++;
			if (max_1 < (int)a1[start][to])
				max_1 = a1[start][to];
			e1--;
		}
	}
	if (end != DELETED) //deleted vertex
	{
		memset(end_deleted, 0, sizeof(u8) * max_e_2);
		for (int i = 0; i < adjList2[end].size(); i++)
		{
			int to = adjList2[end][i];
			if (!this->uG2.flag[to])
			{
				end_deleted[a2[end][to]]++;
				if (max_2 < (int)a2[end][to])
					max_2 = a2[end][to];
				e2--;
			}
		}
	}
	int fi1, fi2;
	int f1, f2;
	int max_idx = max(max_1, max_2);
	for (int i = 0; i <= max_idx; i++)
	{
		fi1 = i < max_e_1 ? this->le1[i] : 0;
		fi2 = i < max_e_2 ? this->le2[i] : 0;

		if (i > max_1)
		{
			f1 = 0;
			f2 = end_deleted[i];
		}
		else if (i > max_2)
		{
			f2 = 0;
			f1 = start_deleted[i];
		}
		else
		{
			f1 = start_deleted[i];
			f2 = end_deleted[i];
		}
		assert(fi1 >= 0 && fi2 >= 0);
		this->updateCommonLabel(fi1, f1, fi2, f2, before_CEL);
	}
	return before_CEL;
}

void treeNode::updateCEL(int &start, int &end, int &e1, int &e2, int &cel)
{
	for (int i = 0; i < adjList1[start].size(); i++)
	{
		int to = adjList1[start][i];
		if (!this->uG1.flag[to])
		{
			this->le1[a1[start][to]]--;
		}
	}
	if (end != DELETED)
	{
		for (int i = 0; i < adjList2[end].size(); i++)
		{
			int to = adjList2[end][i];
			if (!this->uG2.flag[to])
			{
				this->le2[a2[end][to]]--;
			}
		}
	}
	this->CELabel = cel;
	this->uG1.e = e1;
	this->uG2.e = e2;
}

int treeNode::degree_distance_1(int &le1, int &le2, int &v1, int &v2, int &startIndex, int &endIndex,
	vector<int> &vs, int ranki)
{
	int dis = 0, i = 0, size1 = 0, max1 = 0, size2 = 0, max2 = 0, sum = 0, to, mappedVertex;
	this->matching[startIndex] = endIndex;
	for (i = 0; i <= startIndex; i++) //defult order: [0, ..., i]
	{
		mappedVertex = this->matching[i];
		if (mappedVertex == DELETED) dis += succ_degree_1[i];
		else
			dis += abs(succ_degree_1[i] - succ_degree_2[mappedVertex]);
	}
	this->matching[startIndex] = UNMAPPED;

	if (v1 == 0 || v2 == 0)
	{
		le1 = le2 = 0;
		return dis;
	}
	else
	{
		memset(tmpDegree1, 0, max_d_1 * sizeof(u8));
		memset(tmpDegree2, 0, max_d_2 * sizeof(u8));
		for (; i < this->uG1.gs; i++)
		{
			if (max1 < succ_degree_1[i])
				max1 = succ_degree_1[i];
			tmpDegree1[succ_degree_1[i]]++;
		}
		for (i = max1; i >= 0; i--)
		{
			int len = tmpDegree1[i]; //chongdu 
			for (int l = 0; l < len; l++)
				ds1[size1++] = i;
		}
		for (i = 0; i < vs.size(); i++)
		{
			to = vs[i];
			if (to == ranki) continue;
			if (max2 < succ_degree_2[to])
				max2 = succ_degree_2[to];
			tmpDegree2[succ_degree_2[to]]++;
		}
		for (i = max2; i >= 0; i--)
		{
			int len = tmpDegree2[i];
			for (int l = 0; l < len; l++)
				ds2[size2++] = i;
		}
		common::degreeEditDistance(ds1, size1, ds2, size2, le1, le2);
	}
	return dis;
}

int treeNode::edgeOutDistance(int &startIndex, int &endIndex, int &v1, int &v2,
	int &EI, int &ED, int &ES, int &A, int &NA, int &NB)
{
	EI = ED = ES = A = NA = NB = 0;
	if (v1 == 0 || v2 == 0)
	{
		return 0;
	}
	int i = 0, distance = 0;
	int ie = 0, de = 0, mappedVertex;

	memset(unionFlag1, 0, sizeof(bool) * this->uG1.gs);
	memset(unionFlag2, 0, sizeof(bool) * this->uG2.gs);
	this->matching[startIndex] = endIndex;
	this->uG1.flag[startIndex] = true;
	if (endIndex != DELETED)
	{
		this->uG2.flag[endIndex] = true;
	}

	for (; i <= startIndex; i++)
	{
		mappedVertex = this->matching[i];
		if (!succ_degree_1[i] && mappedVertex != DELETED && !succ_degree_2[mappedVertex]) continue;
		if (!succ_degree_1[i] && mappedVertex == DELETED) continue;

		memset(edgeList1, 0, sizeof(u8) * max_e_1);
		memset(edgeList2, 0, sizeof(u8) * max_e_2);
		int maxe1 = -1, maxe2 = -1;
		int e1 = 0, e2 = 0, sum = 0;
		int size = adjList1[i].size();

		for (int j = 0; j < size; j++)
		{
			int to = adjList1[i][j];
			if (!this->uG1.flag[to])
			{
				if (maxe1 < (int)a1[i][to])
					maxe1 = a1[i][to];
				edgeList1[a1[i][to]]++;
				e1++;
				if (!unionFlag1[to])
				{
					ie++;
					unionFlag1[to] = true;
				}
			}
		}
		if (mappedVertex != DELETED)
		{
			size = adjList2[mappedVertex].size();
			for (int j = 0; j < size; j++)
			{
				int to = adjList2[mappedVertex][j];
				if (!this->uG2.flag[to])
				{
					if (maxe2 < (int)a2[mappedVertex][to])
						maxe2 = a2[mappedVertex][to];
					edgeList2[a2[mappedVertex][to]]++;
					e2++;
					if (!unionFlag2[to])
					{
						de++;
						unionFlag2[to] = true;
					}
				}
			}
		}
		int maxe = min(maxe1, maxe2);
		for (int j = 0; j <= maxe; j++)
			sum += min(edgeList1[j], edgeList2[j]);
		distance += max(e1, e2) - sum;
		A += max(e1, e2);
		EI += e1 - sum;
		ED += e2 - sum;
		NA = max(NA, de - ie);
		NB = max(NB, ie - de);
	}

	this->matching[startIndex] = UNMAPPED;
	this->uG1.flag[startIndex] = false;
	if (endIndex != DELETED)
	{
		this->uG2.flag[endIndex] = false;
	}
	return distance;
}

int treeNode::labelEditDistance(int &startIndex, int &endIndex, int &cost, int &v1, int &v2, int &cvl,
	int &e1, int &e2, int &cel, int &bound, bool &flag)
{
	
	int d = cost;	
	int sv = 0, le = 0, dis = 0, ie = 0, de = 0, tmp = 0;
	int edge_1 = 0, edge_2 = 0;
	d += max(v1, v2) - cvl;
	if (d >= bound)
	{
		flag = false;
		return 0;
	}	
	//step 1: \tau \geq max{|V_q|, |V_g|} - |\Sigma_{V_g} \cap \Sigma_{V_q}| + \delta(\sigma_g, \sigma_q)
	memcpy(succ_degree_1, this->degree1, sizeof(u8) * this->uG1.gs);
	memcpy(succ_degree_2, this->degree2, sizeof(u8) * this->uG2.gs);
	this->updateVertexDegree(adjList1, succ_degree_1, startIndex);
	this->updateVertexDegree(adjList2, succ_degree_2, endIndex);
	if (endIndex == DELETED)
		dis = this->degree_distance_1(ie, de, v1, v2, startIndex, endIndex, vs2, -1);
	else
		dis = this->degree_distance_1(ie, de, v1, v2, startIndex, endIndex, vs2, endIndex);

	d += dis;
	d += ie + de; // filter 1 
	if (d >= bound)
	{
		flag = false;
		return 0;
	}

	cel = this->computeCEL(startIndex, endIndex, e1, e2);
	tmp = max(2 * ie + e1 - e2, 2 * de + e2 - e1);
	tmp = max(tmp, max(de, ie + e1 - e2) + e2 - cel);

	d = d - (ie + de);
	d += tmp;

	if (d >= bound)
	{
		flag = false;
		return 0;
	}

	d = d - dis;
	int EI, ED, ES, A, NA, NB, distance;
	distance = this->edgeOutDistance(startIndex, endIndex, v1, v2, EI, ED, ES, A, NA, NB);
	d += distance;
	if (d >= bound)
	{
		flag = false;
		return 0;
	}

	d -= distance;

	tmp = max(EI + NA, ED + NB);
	d += tmp;
	if (d >= bound)
	{
		flag = false;
		return 0;
	}
	else
	{
		flag = true;
		d -= tmp; 
		d += max(distance, tmp);
		return d - cost;
	}
}


void treeNode::generateSuccessors(int &bound, vector<int> &group_1, vector<int> &group_2)
{
	bool flag = false;
	if (this->deep + this->ECost >= bound) return;

	if (uG2.v == 0)
	{
		this->uG1.undealVertexSets(vs1);
		this->ECost = 0;
		int e = 0;
		this->deep += uG1.v;
		if (this->deep >= bound) return;

		for (int j = 0; j < uG1.v; j++)
		{
			int i = vs1[j];
			e += this->getNumberOfAdjacentverifyGraphEdges(this->matching, adjList1, i);
			this->matching[i] = DELETED; // -1 = deletion
		}
		this->deep += e;
		if (this->deep < bound)
		{
			bound = this->deep;
		}
	}
	else if (this->uG1.v == 0)
	{
		this->uG2.undealVertexSets(vs2);
		this->ECost = 0;
		int e = 0;
		this->deep += uG2.v;
		if (this->deep >= bound) return;
		for (int j = 0; j < uG2.v; j++)
		{
			int i = vs2[j];
			e += this->getNumberOfAdjacentverifyGraphEdges(this->inverseMatching, adjList2, i);
			this->inverseMatching[i] = INSERTED; // -2 = insertion
		}
		this->deep += e;
		if (this->deep < bound)
		{
			bound = this->deep;
		}
	}
	else
	{
		this->uG2.undealVertexSets(vs2);
		int rankj = uG1.gs - uG1.v;
		#if 1
		 	if (VERTEXFLAG2) memset(groupFlag2, 0, sizeof(bool) * this->uG2.gs);
		 #endif
		for (int i = 0; i < uG2.v; i++) //the order of A star
		{
			int ranki = i;
			#if 1
			int groupID1 = 0, groupID2 = -1;				
			if (VERTEXFLAG2)
			{
				groupID2 = group_2[vs2[ranki]];
				if (groupFlag2[groupID2]) continue;
				groupFlag2[groupID2] = true;
			}
			if (VERTEXFLAG1)
			{
				groupID1 = group_1[rankj];
				int tmp_gd = this->group[groupID1];
				if (groupID2 < tmp_gd) continue; //here: must optimization with look ahead
			}
			#endif
			verifyGraphNode start, end;
			start = gn1[rankj], end = gn2[vs2[ranki]];
			int cost = this->deep;
			if (start.verifyGraphNodeStr != end.verifyGraphNodeStr) // the verifyGraphNode subtitution
				cost += 1; //substitution

			int startIndex = start.verifyGraphNodeID;
			int endIndex = end.verifyGraphNodeID;
			int cvl = computeCVL(lv1, start.verifyGraphNodeStr, lv2, end.verifyGraphNodeStr);
			cost += processverifyGraphEdges(startIndex, endIndex);
			int v1 = this->uG1.v - 1, v2 = this->uG2.v - 1;
			int e1, e2, cel, estimate_cost;
			estimate_cost = this->labelEditDistance(startIndex, endIndex, cost, v1, v2, cvl, e1, e2, cel, bound, flag);

			if (flag)
			{
				treeNode *tn = new treeNode(*this);//lv1, lv2, le1,  le2, cost, group 
				#if 1
				if (VERTEXFLAG1)
					tn->group[groupID1] = groupID2;
				#endif
				tn->matching[startIndex] = endIndex;
				tn->inverseMatching[endIndex] = startIndex;
				tn->uG1.remove(start, gn1, rankj);
				tn->uG2.remove(end, gn2, vs2[ranki]);
				tn->updateCVL(tn->lv1, start.verifyGraphNodeStr, tn->lv2, end.verifyGraphNodeStr, cvl); //lv1, lv2				
				tn->updateCEL(startIndex, endIndex, e1, e2, cel); //le1, le2

				memcpy(tn->degree1, succ_degree_1, sizeof(u8) * uG1.gs); //degree1
				memcpy(tn->degree2, succ_degree_2, sizeof(u8) * uG2.gs); //degree2
				tn->deep = cost;
				tn->ECost = estimate_cost;
				tn->cost[startIndex] = tn->deep + tn->ECost;
				this->childs.push_back(tn);
			}

		}
		#if 1
		if (uG1.v > uG2.v)
		#endif 
		{
			verifyGraphNode deleted;
			deleted = gn1[rankj];
			int i = deleted.verifyGraphNodeID;
			int cost = this->deep + 1;

			this->matching[i] = DELETED;
			int e = getNumberOfAdjacentverifyGraphEdges(this->matching, adjList1, i);
			this->matching[i] = UNMAPPED;

			cost += e;
			int cvl = computeCVL(lv1, deleted.verifyGraphNodeStr, lv2, DELETED);
			int v1 = this->uG1.v - 1, v2 = this->uG2.v;
			int e1, e2, cel, estimate_cost;
			estimate_cost = labelEditDistance(i, DELETED, cost, v1, v2, cvl, e1, e2, cel, bound, flag);
			if (flag)
			{
				treeNode *tn = new treeNode(*this);//lv1, lv2, le1,  le2, cost, group 
				#if 1
				if (VERTEXFLAG1)
				{
					int groupID1 = group_1[rankj];
					tn->group[groupID1] = 1024;
				}
				#endif
				tn->matching[i] = DELETED;
				tn->uG1.remove(deleted, gn1, rankj);
				tn->updateCVL(tn->lv1, deleted.verifyGraphNodeStr, tn->lv2, DELETED, cvl); //lv1, lv2				
				tn->updateCEL (i, DELETED, e1, e2, cel); //le1, le2

				memcpy(tn->degree1, succ_degree_1, sizeof(u8) * uG1.gs); //degree1
				memcpy(tn->degree2, succ_degree_2, sizeof(u8) * uG2.gs); //degree2
				tn->deep = cost;
				tn->ECost = estimate_cost;
				tn->cost[i] = tn->deep + tn->ECost;
				this->childs.push_back(tn);
			}
		}
	}
}
