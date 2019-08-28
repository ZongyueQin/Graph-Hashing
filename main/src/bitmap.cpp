#include "stdafx.h"
#include "bitmap.h"

bitmap* bitmap::instance = 0;
bitmap* bitmap::getInstance()
{
	if (instance == 0)
	{
		instance = new bitmap();
	}
	return instance;
}
bitmap::bitmap()
{
	this->bmap = new vertexSet[1ULL << 16];
	for (u32 i = 0; i < (1ULL << 16); i++)
	{
		if (i % 2) this->bmap[i].vs.push_back(0);
		for (int idx = 1; idx < 16; idx++)
		{
			u32 value = i >> idx;
			if (value % 2)
				this->bmap[i].vs.push_back(idx);
		}
	}
}
void bitmap::getVertexSet(vector<int> &vs, int &count, u16 &idx, int offset)
{
	int size = this->bmap[idx].vs.size();
	for (int i = 0; i < size; i++)
		vs[i + count] = this->bmap[idx].vs[i] + offset; // all non-zero entry
	count += size;
}

void bitmap::getVertexSet(vector<int> &vs, u64 idx[], int &len)
{
	int size = vs.size();
	if (!size) return;
	
	int count = 0;
	int offset; 
	register u16 r;
	for (int i = 0; i < len; i++)
	{
		offset = i * 64;
		u64 tmp = idx[i];
		r = idx[i] & 0xffff;
		this->getVertexSet(vs, count, r, offset); if (count == size) return;
		r = (idx[i] >> 16) & 0xffff;
		this->getVertexSet(vs, count, r, offset + 16); if (count == size) return;
		r = (idx[i] >> 32) & 0xffff;
		this->getVertexSet(vs, count, r, offset + 32); if (count == size) return;
		r = (idx[i] >> 48) & 0xffff;
		this->getVertexSet(vs, count, r, offset + 48);
	}
}

bitmap::~bitmap()
{
	if (bmap) delete[] bmap;
	bmap = NULL;
}
