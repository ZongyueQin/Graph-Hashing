#pragma once
#ifndef _VERIFY_GRAPH_H
#define _VERIFY_GRAPH_H

#include "graph.h"
#include "bitmap.h"

static bitmap *bm = bitmap::getInstance();

class verifyGraph
{
public:
	int gs;
	int v;
	int e;
	int len;	
	bool *flag;
	u64 unMappedVertex[8];

public:
	verifyGraph(graph &g)
	{
		this->init(g);
	}
	verifyGraph()
	{	
		this->flag = 0;
		gs = v = e = len = 0;
	}
	verifyGraph(const verifyGraph &g)
	{
		*this = g;
	}
	verifyGraph & operator= (const verifyGraph &g)
	{
		if (this != &g)
		{
			this->gs = g.gs;
			this->v = g.v;
			this->e = g.e;
			assert(g.flag);
			this->flag = new bool[gs]; 
			memcpy(this->flag, g.flag, sizeof(bool) * gs);
			
			this->len = g.len;
			memcpy(this->unMappedVertex, g.unMappedVertex, sizeof(u64) * this->len);

		}
		return *this;
	}
	void clear()
	{
		if (this->flag) { delete[] this->flag; this->flag = 0; }
		this->v = this->gs = 0;
	}
	~verifyGraph(){ this->clear(); }

public:
	void init(graph &g)
	{
		this->gs = this->v = g.v;
		this->e = g.e; 
		this->len = this->gs / 64 + (this->gs % 64 ? 1 : 0);
		this->flag = new bool[this->gs]; 
		memset(this->flag, 0, sizeof(bool) * this->gs);
		memset(this->unMappedVertex, 0, sizeof(u64) * this->len);
		
		for (int i = 0; i < len; i++)
		{
			if (i == len - 1) this->unMappedVertex[i] = (1ULL << (this->gs - i * 64)) - 1;
			else this->unMappedVertex[i] = (1ULL << 64) - 1;
		}
	}
	void remove(verifyGraphNode &node, verifyGraphNode *gn, const int &pos)
	{
		assert(pos >= 0 && pos < this->gs && this->flag && !flag[pos]);
		node = gn[pos];
		this->flag[pos] = true;
		this->v--;
		int offset = pos / 64;
		this->unMappedVertex[offset] = this->unMappedVertex[offset] - (1ULL << (pos - offset * 64));
	}
public:
	void undealVertexSets(vector<int> &vs)
	{
		vs.clear();
		if (!this->v) return;
		vs.resize(this->v, 0);
		bm->getVertexSet(vs, this->unMappedVertex, this->len);
	}

};

#endif
