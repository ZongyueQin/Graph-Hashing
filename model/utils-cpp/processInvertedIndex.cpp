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
	size_t tupleCnt = 0;

	fin >> totalCnt >> embLen;

	InvertedIndexEntry *invertedIndex = new InvertedIndexEntry[totalCnt];
	
	for(int i = 0; i < totalCnt; i++)
	{
		fin >> invertedIndex[i].code;
		int len;
		fin >> len;
		invertedIndex[i].infos.resize(len);
		tupleCnt += len;
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
	if (fd1 == -1)
	{
		fprintf(stderr, "opem: %s\n", strerror(errno));
		return -1;
	}
	ftruncate(fd1, sizeof(CodePos)*totalCnt);

	CodePos *code2Pos = (CodePos*) mmap(NULL, sizeof(CodePos)*totalCnt, 
					PROT_WRITE|PROT_READ, 
					MAP_SHARED,
					fd1, 0);
	if (code2Pos == (void *)-1)
	{
		fprintf(stderr, "mmap: %s\n", strerror(errno));
		return -1;
	}
	fprintf(stderr, "mmap succeed\n");

	close(fd1);

	int fd2 = open(argv[3], O_CREAT|O_RDWR|O_TRUNC, 00777);
	if (fd2 == -1)
	{
		fprintf(stderr, "opem: %s\n", strerror(errno));
		return -1;
	}
	ftruncate(fd2, tupleCnt*sizeof(GInfo));

	GInfo *invertedIndexValue = (GInfo*) mmap(NULL, tupleCnt*sizeof(GInfo),
						PROT_WRITE|PROT_READ,
						MAP_SHARED,
						fd2, 0);
	if (invertedIndexValue == (void*)-1)
	{
		fprintf(stderr, "mmap2: %s\n", strerror(errno));
		return -1;
	}
	close(fd2);
	fprintf(stderr, "mmap succeed\n");


	int pos = 0;
	for(int i = 0; i < totalCnt; i++)
	{
//		fprintf(stderr, "%d\n", i);
		(code2Pos+i)->code = invertedIndex[i].code;
		(code2Pos+i)->pos = pos;
//		fprintf(stderr, "code2pos succeed\n");
		for(int j = 0; j < invertedIndex[i].infos.size(); j++)
		{
			invertedIndexValue[pos] = invertedIndex[i].infos[j];
			pos++;
		}
	}

	fprintf(stderr, "write succees\n");
	fprintf(stdout, "total code count: %d\n", totalCnt);
	fprintf(stdout, "total graph count: %d\n", tupleCnt);
	fprintf(stdout, "%d\n", code2Pos->code);

	munmap((void*) code2Pos, sizeof(CodePos)*totalCnt);
	munmap((void*) invertedIndexValue, tupleCnt * sizeof(GInfo));
	return 0;
}
