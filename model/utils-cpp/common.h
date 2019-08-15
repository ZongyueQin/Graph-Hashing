#include <fstream>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <algorithm>
#include <vector>
#include<sys/types.h>
#include<fcntl.h>
#include<string.h>
struct CodePos
{
	int code;
	int pos;
};

class GInfo
{
public:
	uint64_t gid;
	double emb[300];
	GInfo()
	{
		gid = 0;
		memset(emb, 0, sizeof(emb));
	}
	GInfo(const GInfo& a)
	{
		gid = a.gid;
		for(int i = 0; i < 300; i++)
			emb[i] = a.emb[i];
	}
	GInfo& operator = (const GInfo &a)
	{
		gid = a.gid;
		for(int i = 0; i < 300; i++)
			emb[i] = a.emb[i];
		return *this;	
	}
};

class InvertedIndexEntry
{
public:
	uint64_t code;
	std::vector <GInfo> infos;
};

bool operator < (const InvertedIndexEntry &a, const InvertedIndexEntry &b)
{
	return a.code < b.code;
}
