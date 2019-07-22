#pragma once 
#ifndef _BITMAP_H
#define _BITMAP_H
#include "stdafx.h"

typedef struct vertexSet
{
	vector<int> vs;
	vertexSet(){ vs.clear(); }
	~vertexSet(){ vector<int>().swap(vs); }
};

class bitmap
{
private:
	vertexSet *bmap;
	static bitmap* instance;
	void getVertexSet(vector<int> &vs, int &count, u16 &idx, int offset);
public:
	static bitmap* getInstance();
	void getVertexSet(vector<int> &vs, u64 idx[], int &len);
	~bitmap();
private:
	bitmap();
};
#endif