#include "stdafx.h"
#include "graph.h"

map<int, int> graph::vertexLabel(int &max_vertex)
{
	max_vertex = 0;
	map<int, int> mvl;
	map<int, int>::iterator iter;
	for (int i = 0; i < v; i++)
	{
		int str = V[i];
		iter = mvl.find(str);
		if (iter == mvl.end())
		{
			mvl.insert(pair<int, int>(str, 1));
			if (max_vertex < str)
				max_vertex = str;
		}			
		else iter->second++;
	}
	return mvl;
}

map<int, int> graph::edgeLabel(int &max_edge)
{
	max_edge = 0; 
	map<int, int> mel;
	map<int, int>::iterator iter;
	for (int j = 0; j < v; j++)
	{
		for (int i = j + 1; i < v; i++)
		{
			if (this->E[j][i] != 255)
			{
				int einfo = this->E[j][i];
				iter = mel.find(einfo);
				if (iter == mel.end())
				{
					mel.insert(pair<int, int>(einfo, 1));
					if (max_edge < einfo) max_edge = einfo;
				}
				else iter->second++;
			}
		}
	}
	return mel;
}
void graph::vertexDegree(int &vertex, u8 &degree)
{
	degree = 0;
	for (int i = 0; i < this->v; i++)
	{
		if (vertex == i) continue;
		if (this->E[i][vertex] != 0xff)
			degree++;
	}
}
void graph::vertexDegree(int &vertex, int &degree)
{
	degree = 0;
	for (int i = 0; i < this->v; i++)
	{
		if (vertex == i) continue;
		if (this->E[i][vertex] != 0xff)
			degree++;
	}
}
void graph::editDistanceInduced(graph &g, graph &h, vector<vector<int > >&assignment, int &value)
{

	int VI, VD, VS, EI, ES, ED;  VI = VD = VS = EI = ES = ED = 0;
	value = 0;
	vector<int> matching; matching.resize(assignment.size(), 255);

	for (int i = 0; i < assignment.size(); i++)
	{
		for (int j = 0; j < assignment.size(); j++)
		{
			if (assignment[i][j] != 0)
			{
				if (j < h.v) matching[i] = j;
				else
					matching[i] = 255;
			}
		}

	}
	for (int i = 0; i < g.v; i++)
	{
		if (matching[i] == 255)
			VD++;
		else if (g.V[i] != h.V[matching[i]])
			VS++;
		else
			;
	}
	for (int i = 0; i < g.v; i++)
	{
		int mappedVertex = matching[i];
		if (mappedVertex != 255)
		{
			for (int j = i + 1; j < g.v; j++)
			{
				int mappedVertexTwo = matching[j];
				if (g.E[i][j] != 255)
				{
					if (mappedVertexTwo != 255)
					{
						if (h.E[mappedVertex][mappedVertexTwo] == 255)
							ED++;
						else if (h.E[mappedVertex][mappedVertexTwo] != g.E[i][j])
							ES++;
						else
							;
					}
				}
				else
				{
					if (mappedVertexTwo != 255 && h.E[mappedVertex][mappedVertexTwo] != 255)
						EI++;
				}
			}
		}
		else
		{
			for (int j = i + 1; j < g.v; j++)
			{
				if (g.E[i][j] != 255)
					ED++;
			}
		}
	}
	value = EI + ED + ES + VD + VI + VS;
}

inline vector<BTuple> graph::neibhoor(int &id)
{
	vector<BTuple> vp;
	for (int i = 0; i < this->v; i++)
	{
		if (this->E[id][i] != 255)
		{
			vp.push_back(BTuple(i, this->E[id][i]));
		}
	}
	return vp;
}

bool graph::equal(int &f, int &t)
{
	if (V[f] != V[t]) return false;
	vector<BTuple > vf = neibhoor(f);
	vector<BTuple > vt = neibhoor(t);
	int size1 = vf.size(), size2 = vt.size();
	if (size1 != size2) return false;
	sort(vf.begin(), vf.end());
	sort(vt.begin(), vt.end());
	int i = 0;
	while (i < size1)
	{
		BTuple t1 = vf[i], t2 = vt[i];
		if (t1.first == t2.first)
		{
			if (t1.second == t2.second) { i++; continue; }
			else 
				return false;
		}
		else 
			return false;
	}
	return true;
}

bool inVector(vector<int> &vr, int value)
{
	int size = vr.size();
	for (int i = 0; i < size; i++)
	{
		if (vr[i] == value) return true;
	}
	return false;
}

vector<BTuple> graph::equalSet()
{
	vector<BTuple> tmp;
	for (int i = 0; i < v; i++)
	{
		for (int j = i + 1; j < v; j++)
		{
			if (this->equal(i, j))
				tmp.push_back(BTuple(i, j));
		}
	}
	return tmp;
}

void graph::mergeEqualSet(vector<BTuple> &vb, vector<vector<int> > &vertexEqual)
{
	vertexEqual.clear();
	int size = vb.size();
	if (!size) return;
	vector<bool> flag(size, false);
	for (int i = 0; i < size; i++)
	{
		if (flag[i]) continue;
		flag[i] = true;
		vector<int> tmp;
		tmp.push_back(vb[i].first); 
		tmp.push_back(vb[i].second);
		for (int j = i + 1; j < size; j++)
		{
			if (inVector(tmp, vb[j].first) || inVector(tmp, vb[j].second)) //mergeSet 
			{
				flag[j] = true;
				if (!inVector(tmp, vb[j].first))tmp.push_back(vb[j].first);
				if (!inVector(tmp, vb[j].second))tmp.push_back(vb[j].second);
			}
		}
		vertexEqual.push_back(tmp);
	}
}
u64  graph::divideGroup(vector<int> &vs, int &group_number)
{
	u64 s = 1;
	vector<BTuple> vb = this->equalSet(); vector<vector<int> > vertexEqual;
	this->mergeEqualSet(vb, vertexEqual);

	vs.resize(v, -1); int idx = 0;
	int ves = vertexEqual.size();
	if (ves == 0)
	{
		group_number = 0;
		for (int i = 0; i < v; i++)
			vs[i] = idx++;
		return s;
	}
	for (int i = 0; i < ves; i++)
	{
		int size = vertexEqual[i].size();
		for (int j = 0; j < size; j++)
		{
			vs[vertexEqual[i][j]] = idx;
		}
		idx++;
		s = s * size;
	}
	for (int i = 0; i < v; i++)
	{
		if (vs[i] == -1) vs[i] = idx++;
	}
	group_number = idx;
	return s;
}

void graph::DFS(int &i, int &order_count, vector<bool> &visited, vector<int> &vertex_order, int *vertex_degree)
{
	int j = 0;
	visited[i] = 1;
	vertex_order[order_count++] = i;

	int len = 0;
	for(int j = 0; j < v; j++)
	{
		if(E[i][j] != 0xff && !visited[j])
		{
			order_tmp[len].first = vertex_degree[j];
			order_tmp[len].second = j;
			len++;
		}
	}	
	if (len > 0)
	{
		sort(order_tmp, order_tmp + len);
		for (int i = 0; i < len; i++)
		{			
			j = order_tmp[i].second;
			if (!visited[j])
				DFS(j, order_count, visited, vertex_order, vertex_degree);
		}

	}
}
void graph::DFSTraverse(vector<int> &vertex_order)
{
	vector<bool> visited(v, false);
	int i = 0, order_count = 0;
	vector<int> tmp(v, -1);

	int max_d;
	this->degreeSet(vertex_degree, max_d);

	for (int i = 0; i < this->v; i++)
	{
		order_vertex[i].first = vertex_degree[i];
		order_vertex[i].second = i;
	}
	sort(order_vertex, order_vertex + v);

	for (i = 0; i < v; i++)
	{
		int j = order_vertex[i].second;
		if (!visited[j])
			DFS(j, order_count, visited, tmp, vertex_degree);
	}
	vertex_order.clear(); vertex_order.resize(v, -1);
	for (int i = 0; i < v; i++)
		vertex_order[tmp[i]] = i;
}