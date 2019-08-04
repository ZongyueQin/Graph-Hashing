#include <fstream>
#include <assert.h>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <cstdio>
#include <errno.h>
#include <algorithm>
#include <vector>
#include "common.h"
#include<sys/types.h>
#include<fcntl.h>
#include <unistd.h>
#include <string.h>

using namespace std;

/*struct GInfo
{
	uint64_t gid;
	double emb[64];
};

class InvertedIndexEntry
{
public:
	uint64_t code;
	vector <GInfo> infos;
};

bool operator < (const InvertedIndexEntry &a, const InvertedIndexEntry &b)
{
	return a.code < b.code;
}*/

int main(int argc, char **argv)
{
	assert(argc == 4);
	string filename = argv[1];
	ifstream fin(filename);
	if (!fin)
	{
		printf("Failed to open %s\n", argv[1]);
		return -1;
	}	
	
	int totalCnt, embLen;
	size_t fileLen = 0;

	fin >> totalCnt >> embLen;

	InvertedIndexEntry *invertedIndex = new InvertedIndexEntry[totalCnt];
	
	for(int i = 0; i < totalCnt; i++)
	{
		fin >> invertedIndex[i].code;
		int len;
		fin >> len;
		invertedIndex[i].infos.resize(len);
		fileLen += sizeof(GInfo) * len;
		for(int j = 0; j < len; j++)
		{
			fin >> invertedIndex[i].infos[j].gid;
			for(int k = 0; k < embLen; k++)
			{
				fin >> invertedIndex[i].infos[j].emb[k];
			}
		}
	}

	sort(invertedIndex, invertedIndex + totalCnt);
	
	int fd1 = open(argv[2], O_CREAT|O_RDWR|O_TRUNC, 00777);

	int *code2Pos = (int*) mmap(NULL, 2*sizeof(uint64_t)*totalCnt, 
					PROT_WRITE, 
					MAP_SHARED,
					fd1, 0);
	if (code2Pos == (void *)-1)
	{
		fprintf(stderr, "mmap: %s\n", strerror(errno));
		return -1;
	}

	close(fd1);

	int fd2 = open(argv[3], O_CREAT|O_RDWR|O_TRUNC, 00777);
	GInfo *invertedIndexValue = (GInfo*) mmap(NULL, fileLen,
						PROT_WRITE,
						MAP_PRIVATE|MAP_LOCKED,
						fd2, 0);
	if (invertedIndexValue == (void*)-1)
	{
		fprintf(stderr, "mmap2: %s\n", strerror(errno));
		return -1;
	}
	close(fd2);

	int pos = 0;
	for(int i = 0; i < totalCnt; i++)
	{
		code2Pos[2*i] = invertedIndex[i].code;
		code2Pos[2*i+1] = pos;
		for(int j = 0; j < invertedIndex[i].infos.size(); j++)
		{
			invertedIndexValue[pos] = invertedIndex[i].infos[j];
			pos++;
		}
	}

	munmap((void*) code2Pos, 2*sizeof(uint64_t)*totalCnt);
	munmap((void*) invertedIndexValue, fileLen);
	return 0;
}
