#pragma once
#ifndef _GRAPH_H
#define _GRAPH_H
#include "common.h"

struct BTuple
{
	int first;
	int second;
	BTuple(){}
	BTuple(int f, int s) :first(f), second(s){}
	bool operator < (const BTuple &t) const
	{ return first < t.first || (first == t.first && second < t.second);}
};

//extra variable used for DFS traversal 
static BTuple order_tmp[512];
static BTuple order_vertex[512];
static int vertex_degree[512];

class graph
{
public:
	int v;
	vector<int> V;
	vector<vector<int > > E;
	int graph_id;
	int e;

public:
	graph(){}
	~graph(){}
	inline int edgeinfo(int from, int to)
	{
		assert(from < v && to < v);
		return E[from][to];
	}
	static void reOrderGraphs(const char *in, const char *out, int total)
	{	
		FILE *fr = fopen(in, "r"); assert(fr);
		FILE *fw = fopen(out, "w"); assert(fw);

		int gid, v, e;
		int f, t, l;
		int count = 0;

		while (!feof(fr))
		{
			fscanf(fr, "%d\n", &gid);
			fscanf(fr, "%d %d\n", &v, &e);
			fprintf(fw, "%d\n", gid);
			fprintf(fw, "%d %d\n", v, e);

			graph g;
			g.graph_id = gid; g.v = v; g.e = e;
			g.V.resize(g.v, 0); vector<int> tmp(g.v, 255); g.E.resize(g.v, tmp);
			vector<int> from, to, label; from.resize(g.e, -1); 
			to.resize(g.e, -1); label.resize(g.e, -1);

			for (int i = 0; i < v; i++)
				fscanf(fr, "%d\n", &g.V[i]);
			vector<int> vertex_tmp = g.V;

			for (int i = 0; i < e; i++)
			{
				fscanf(fr, "%d %d %d\n", &f, &t, &l);
				g.E[f][t] = l;
				g.E[t][f] = l;
				from[i] = f;
				to[i] = t;
				label[i] = l;
			}		
			vector<int> vertex_order;
			g.DFSTraverse(vertex_order);

			for (int i = 0; i < v; i++)
			{
				int rank = vertex_order[i];
				vertex_tmp[rank] = g.V[i];
			}
			for (int i = 0; i < v; i++)
			{
				fprintf(fw, "%d\n", vertex_tmp[i]);
			}
			for (int i = 0; i < e; i++)
			{
				int rankf = vertex_order[from[i]];
				int rankt = vertex_order[to[i]];
				if (rankt < rankf)
				{
					int tmp = rankt;
					rankt = rankf;
					rankf = tmp;
				}
				fprintf(fw, "%d %d %d\n", rankf, rankt, label[i]);
			}
			count++;
			if (count >= total) break;
		}
		if (fr) fclose(fr);
		if (fw) fclose(fw);
	}

	static vector<graph> readGraphMemory(const char *db, int total)
	{
		vector<graph> vg;
		FILE *fr = fopen(db, "r+"); assert(fr);
		int v, e;
		int gid;
		int f, t, l;
		int count = 0;

		while (!feof(fr))
		{
			fscanf(fr, "%d\n", &gid);
			fscanf(fr, "%d %d\n", &v, &e);
			graph g;
			g.graph_id = gid; g.v = v; g.e = e;
			g.V.resize(g.v, 0); vector<int> tmp(g.v, 255); g.E.resize(g.v, tmp);

			for (int i = 0; i < v; i++)
				fscanf(fr, "%d\n", &g.V[i]);

			for (int i = 0; i < e; i++)
			{
				fscanf(fr, "%d %d %d\n", &f, &t, &l);
				g.E[f][t] = l;
				g.E[t][f] = l;
			}
			vg.push_back(g);
			count++;
			if (count >= total)
				break;
		}
		if (fr) fclose(fr);
		return vg;
	}
	graph(const graph &g)
	{
		this->graph_id = g.graph_id;
		this->v = g.v;
		this->e = g.e;
		this->V = g.V;
		this->E = g.E;
	}
public:
	map<int, int> vertexLabel(int &max_vertex);
	map<int, int> edgeLabel(int &max_edge);
	void vertexDegree(int &vertex, u8 &degree);
	void vertexDegree(int &vertex, int &degree);
	void editDistanceInduced(graph &g, graph &h, vector<vector<int > >&assignment, int &value);
	inline void degreeSet(int* vd, int &max_d)
	{
		max_d = 0;
		for (int i = 0; i < this->v; i++)
		{
			this->vertexDegree(i, vd[i]);
			if (max_d < vd[i])
				max_d = vd[i];
		}
	}
	inline void degreeSet(u8 *vd, int &max_d)
	{
		max_d = 0;
		for (int i = 0; i < this->v; i++)
		{
			this->vertexDegree(i, vd[i]);
			if (max_d < vd[i])
				max_d = vd[i];
		}
	}	
public:
	inline vector<BTuple > neibhoor(int &id);
	bool equal(int &f, int &t);
	vector<BTuple> equalSet();
	void mergeEqualSet(vector<BTuple> &vb, vector<vector<int> > &vertexEqual);
	u64  divideGroup(vector<int> &vs, int &group_number);
public:
	void DFS(int &i, int &order_count, vector<bool> &visited, vector<int> &vertex_order, int *degree);
	void DFSTraverse(vector<int> &vertex_order);
};
#endif
