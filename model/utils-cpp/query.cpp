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
#include <queue>


using namespace std;

class SearchNode
{
public:
	uint64_t code;
	int dist;
	int last_flip_pos;
	SearchNode(uint64_t c, int d, int l)
	{
		code = c; dist=d; last_flip_pos = l;
	}
};

double dist(GInfo &a, GInfo &b, int embLen)
{
	double ret = 0;
	for(int i = 0; i < embLen; i++)
	{
		ret += (a.emb[i]-b.emb[i])*(a.emb[i]-b.emb[i]);
	}
	return ret;
}

int BinarySearch(CodePos *array, int len, uint64_t code)
{
	int left = 0;
	int right = len - 1;
	while (right >= left)
	{
		int mid = (right+left)/2;
		if (array[mid].code == code)
		{
			return mid;
		}
		else if (array[mid].code < code)
		{
			left = mid + 1;
		}
		else
		{
			right = mid - 1;
		}
	}
	return -1;
}

// return is in res, the index (position) in Code2Pos
void getAllValidCode(uint64_t code, int thres, int totalCodeCnt, int embLen, CodePos *index, vector<uint64_t> &res)
{
	queue <SearchNode> que;
	que.push(SearchNode(code, 0, -1));
	while (!que.empty())
	{
		SearchNode curNode = que.front();
		que.pop();
        	
		int idx = BinarySearch(index, totalCodeCnt, curNode.code);
		if (idx > -1)
		{
			res.push_back(idx);
//			fprintf(stdout, "%d\n", index[idx].code);
		}
#ifdef LOOSE
		if (curNode.dist <= thres)
#else
		if (curNode.dist < thres)
#endif
		{
			int pow = curNode.last_flip_pos + 1;
			for(; pow < embLen; pow++)
			{
				uint64_t mask = (1 << pow);
				uint64_t newCode = curNode.code ^ mask;
				que.push(SearchNode(newCode, curNode.dist+1, pow));
			}			
		}
	}
    
} 


int main(int argc, char **argv)
{
	// ./query code thres file1 file2 totalCodeCnt totalGraphCnt codeLen embLen fine-grained embedding
	if (argc <= 9)
	{
		fprintf(stdout, "command format:./query code search_thres file1 file2 totalCodeCnt totalGraphCnt codeLen embLen fine-grained fine_grained_thres [embedding]\n");
		return -1;
	}
	uint64_t queryCode = atoi(argv[1]);
	int thres = atoi(argv[2]);	

	
	int fd1 = open(argv[3], O_RDWR, 00777);

	int totalCodeCnt = atoi(argv[5]);
	CodePos *code2Pos = (CodePos*) mmap(NULL, 2*sizeof(uint64_t)*totalCodeCnt, 
					PROT_READ, 
					MAP_SHARED,
					fd1, 0);
	if (code2Pos == (void *)-1)
	{
		fprintf(stderr, "mmap: %s\n", strerror(errno));
		return -1;
	}

	close(fd1);

	int totalGraphCnt = atoi(argv[6]);
	int fd2 = open(argv[4], O_RDWR, 00777);
	GInfo *invertedIndexValue = (GInfo*) mmap(NULL, totalGraphCnt*sizeof(GInfo),
						PROT_WRITE,
						MAP_PRIVATE|MAP_LOCKED,
						fd2, 0);
	if (invertedIndexValue == (void*)-1)
	{
		fprintf(stderr, "mmap2: %s\n", strerror(errno));
		return -1;
	}
	close(fd2);

        int codeLen = atoi(argv[7]);
	int embLen = atoi(argv[8]); 
	GInfo qInfo;
	int fine_grained = atoi(argv[9]);
        int fine_grained_thres = 0;
	if (fine_grained > 0)
	{
                fine_grained_thres = atoi(argv[10]);
		for(int i = 0; i < embLen; i++)
		{
			qInfo.emb[i] = atof(argv[i+11]);
		}
	}

	vector <uint64_t> validCode;
	getAllValidCode(queryCode, thres, totalCodeCnt, codeLen, code2Pos, validCode);

	for(int i = 0; i < validCode.size(); i++)
	{
		int start = code2Pos[validCode[i]].pos;
		int end = code2Pos[validCode[i]+1].pos;
		for(int j = start; j < end; j++)
		{
			if (fine_grained > 0 && dist(qInfo, invertedIndexValue[j], embLen) > (double)fine_grained_thres)
			{
				continue;
			}
			printf("%ld\n", invertedIndexValue[j].gid);
		}
	}

	munmap((void*) code2Pos, 2*sizeof(uint64_t)*totalCodeCnt);
	munmap((void*) invertedIndexValue, sizeof(GInfo)*totalGraphCnt);
	return 0;
}
