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

struct CodePos
{
	int code;
	int pos;
};

struct GInfo
{
	uint64_t gid;
	double emb[64];
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
