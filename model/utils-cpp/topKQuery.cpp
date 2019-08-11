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

double dist(const GInfo &a, const GInfo &b, int embLen)
{
	double ret = 0;
	for(int i = 0; i < embLen; i++)
	{
		ret += (a.emb[i]-b.emb[i])*(a.emb[i]-b.emb[i]);
	}
	return ret;
}

class Comp
{
public:
	static GInfo qInfo;
	static int embLen;
	bool operator() (const GInfo &a, const GInfo &b)
	{
		return dist(a, qInfo, embLen) < dist(b, qInfo, embLen);
	}	
};
GInfo Comp::qInfo = GInfo();
int Comp::embLen = 0;

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

void getTopKByEmb(int K, int graphCnt, GInfo *gInfo, vector<int> gid)
{
	priority_queue<GInfo, vector<GInfo>, Comp> heap;
	for(int i = 0; i < graphCnt; i++)
	{
		heap.push(gInfo[i]);
		if (heap.size() > K)
			heap.pop();
	}	
	while (!heap.empty())
	{
		gid.push_back(heap.top().gid);
		heap.pop();
	}
}

int getTopKByCode(int K, uint64_t code, int codeLen, int thres, int totalCodeCnt, int embLen, 
            int totalGraphCnt, CodePos *index, vector<uint64_t> &res)
{
	queue <SearchNode> que;
	que.push(SearchNode(code, 0, -1));
        int curHDist = 0;
        uint64_t num2Search = 1;
	int candNum = 0;
	while (!que.empty() && candNum < K)
	{
		SearchNode curNode = que.front();
		que.pop();
        	
		if (curNode.dist > curHDist)
		{
			num2Search *= (codeLen-curHDist);
			curHDist = curNode.dist;
			num2Search /= curHDist;
			if (num2Search > totalGraphCnt)
			{
				// Search cost too much, return -1 to change strategy
				return -1;
			}
		}

		int idx = BinarySearch(index, totalCodeCnt, curNode.code);
		if (idx > -1)
		{
			res.push_back(idx);
			int graphNum = 0;
			if (idx == totalCodeCnt - 1)
				graphNum = totalGraphCnt - index[idx].pos;
			else
				graphNum = index[idx+1].pos - index[idx].pos;
			 candNum += graphNum;
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
	return candNum;
} 


int main(int argc, char **argv)
{
	// ./query code thres file1 file2 totalCodeCnt totalGraphCnt codeLen embLen fine-grained embedding
	if (argc <= 9)
	{
		fprintf(stdout, "command format:./query code search_thres file1 file2 totalCodeCnt totalGraphCnt codeLen embLen K [embedding]\n");
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
        int K = atoi(argv[9]);
	for(int i = 0; i < embLen; i++)
	{
		qInfo.emb[i] = atof(argv[i+10]);
	}

	Comp::embLen = embLen;
	Comp::qInfo = qInfo;

	vector <uint64_t> validCode;
	vector<int> gids;
	int ret = getTopKByCode(K, queryCode, codeLen, thres, totalCodeCnt, embLen, totalGraphCnt, code2Pos, validCode);
	if (ret > 0)
	{
		GInfo *candidates = new GInfo[ret];
		int pos = 0;
		for(int i = 0; i < validCode.size(); i++)
		{
			int start = code2Pos[validCode[i]].pos;
			int end = code2Pos[validCode[i]+1].pos;
			for(int j = start; j < end; j++)
			{
				candidates[pos] = invertedIndexValue[j];
				pos++;
//				printf("%ld\n", invertedIndexValue[j].gid);
			}
		}
		getTopKByEmb(K, ret, candidates, gids);
	}
	else
	{
		getTopKByEmb(K, totalGraphCnt, invertedIndexValue, gids);
	}

	assert(gids.size() == K);
	for(int i = 0; i < K; i++)
	{
		printf("%ld\n", gids[i]);
	}

	munmap((void*) code2Pos, 2*sizeof(uint64_t)*totalCodeCnt);
	munmap((void*) invertedIndexValue, sizeof(GInfo)*totalGraphCnt);
	return 0;
}
