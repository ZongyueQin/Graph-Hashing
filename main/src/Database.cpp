#include "Database.h"

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
		//cout << i << ' ';
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

uint64_t Combinations(unsigned int n, int k)
{
     if (k > n)
         return 0;
     if (k < 0)
        return 0;

     uint64_t r = 1;
     k = k < n-k ? k : n-k;
     for (unsigned int d = 1; d <= (unsigned)k; ++d)
     {
         r *= (double)n;
         n--;
         r /= d;
     }
     return r;
}

uint64_t computeSearchSpace(int codeLen, int thres)
{
	uint64_t ret = 0;
	for(int i = 0; i <= thres; i++)
		ret += Combinations(codeLen, i);
	return ret;
}

uint64_t getHammingDistance(uint64_t a, uint64_t b, int len)
{
	uint64_t diff= a ^ b;
	uint64_t ret = 0;
	for(int i = 0; i < len; i++)
	{
		uint64_t bit = ((diff >> i) & 0x1);
		ret += bit;
	}
	return ret;
}

// return is in res, the index (position) in Code2Pos
void getAllValidCode(uint64_t code, int thres, 
				int totalCodeCnt, int codeLen, 
				CodePos *index, vector<uint64_t> &res)
{
	uint64_t searchSpace = computeSearchSpace(codeLen, thres);
	if (searchSpace > totalCodeCnt)
	{
		for(int i = 0; i < totalCodeCnt; i++)
		{
			uint64_t newCode = index[i].code;
			uint64_t hammingDistance = getHammingDistance(newCode, code, codeLen);
			if (hammingDistance <= (uint64_t)thres)
				res.push_back(i);
		}
		return;
	
	}

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
			for(; pow < codeLen; pow++)
			{
				uint64_t mask = (1 << pow);
				uint64_t newCode = curNode.code ^ mask;
				que.push(SearchNode(newCode, curNode.dist+1, 
							pow));
			}			
		}
	}
    
} 

class CompForTopK
{
public:
	static GInfo qInfo;
	static int embLen;
	bool operator() (const GInfo &a, const GInfo &b)
	{
		return dist(a, qInfo, embLen) < dist(b, qInfo, embLen);
	}	
};
GInfo CompForTopK::qInfo = GInfo();
int CompForTopK::embLen = 0;



void getTopKByEmb(int K, int graphCnt, GInfo *gInfo, deque<uint64_t>& gid)
{
	priority_queue<GInfo, vector<GInfo>, CompForTopK> heap;
	for(int i = 0; i < graphCnt; i++)
	{
		heap.push(gInfo[i]);
		if (heap.size() > K)
			heap.pop();
	}	
	while (!heap.empty())
	{
		gid.push_front(heap.top().gid);
		heap.pop();
	}
}


int getTopKByCode(int K, uint64_t code, int codeLen, int thres, int totalCodeCnt, 
            int totalGraphCnt, CodePos *index, vector<uint64_t> &res)
{
	queue <SearchNode> que;
	que.push(SearchNode(code, 0, -1));
        int curHDist = 0;
        uint64_t num2Search = 1;
	int candNum = 0;
        int cnt = 0;
	while ((!que.empty()) && candNum < 5*K)
	{
		SearchNode curNode = que.front();

		que.pop();
		if (curNode.dist > curHDist)
		{
			cnt = 0;
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
			for(; pow < codeLen; pow++)
			{
				uint64_t mask = (1 << pow);
				uint64_t newCode = curNode.code ^ mask;
				que.push(SearchNode(newCode, curNode.dist+1, pow));
			}			
		}
	}
	return candNum;
}
/**
 Construtor
 @param model_path: the path to load model
 @db: the path to load database graphs
 @graphNum: how many graphs to load
 */
Database::Database(const string &modelPath, const string &db, 
			const string &invIndex, 
			int totalGraph, int _codeLen,
			const string &GED2HammingFile)
{
	if (GED2HammingFile == "")
		for(int i = 0; i < 10; i++)
			GED2Hamming[i] = i + 1;
			//GED2Hamming[i] = i + 2;
	else
	{
		ifstream ged2HamFin(GED2HammingFile);
		if (!ged2HamFin)
		{
			cout << "Failed to open " << GED2HammingFile << endl;
			exit(0);
		}
		for(int i = 0; i < 10; i++)
		{
			int ham;
			ged2HamFin >> ham;
			GED2Hamming[i] = ham;
		}
	}

	codeLen = _codeLen;
	totalEncodeTime = 0;
	totalSearchTime = 0;
	totalVerifyTime = 0;

	/* load python module */
	Py_Initialize();

	string chdir_cmd = string("sys.path.append(\"") + pythonPath + "\")";
	const char* cstr_cmd = chdir_cmd.c_str();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);

	PyObject *moduleName = PyUnicode_FromString("coreFunc");
	PyObject *pModule = PyImport_Import(moduleName);
	if (!pModule)
	{
		PyRun_SimpleString("print('')");
		exit(0);
	}
	cout << "[INFO] Python get module succeed" << endl;


	/* Load python tensorflow model */
	PyObject *LoadModel = PyObject_GetAttrString(pModule, "loadModel");
	if (!LoadModel || !PyCallable_Check(LoadModel))
	{
		cout << "[ERROR] Can't find function (loadModel)" << endl;
		exit(0);
	}	
	cout << "[INFO] Get function (loadModel) succeed" << endl;

	PyObject *py_model_path	= PyUnicode_FromString(modelPath.c_str());
	PyObject *args = PyTuple_New(1);
	PyTuple_SetItem(args, 0, py_model_path);
	PyObject* pRet = PyObject_CallObject(LoadModel, args); 	
	if (!pRet)
	{
		cout << "Failed to load model from " << modelPath << endl;
		PyRun_SimpleString("print('')");
		exit(0);
	}

	getCodeAndEmbByQid = PyObject_GetAttrString(pModule, 
							"getCodeAndEmbByQid");
	if (!getCodeAndEmbByQid || !PyCallable_Check(getCodeAndEmbByQid))
	{
		cout << "[ERROR] Can't find function (getCodeAndEmbByQid)" 
			<< endl;
		exit(0);
	}
	cout << "[INFO] Get function (getCodeAndEmbByGid) succeed" << endl;
	

	/* load databases */
	cout << "Loading graphs..." << endl;
	string db_out = db+"_ordered";
	graph::reOrderGraphs(db.c_str(), db_out.c_str(), totalGraph);
	graphDB = graph::readGraphMemory(db_out.c_str(), totalGraph);
	std::remove(db_out.c_str()); 
	for(int i = 0; i < graphDB.size(); i++)
		gid2Pos.insert(make_pair(graphDB[i].graph_id, i));

	/* load inverted index file */

	cout << "Finish loading graphs" << endl; 
	cout << "Loading inverted index" << endl;
	ifstream fin(invIndex);
	if (!fin)
	{
		cout << "Failed to open " << invIndex << endl;
		exit(0);
	}	
	
	fin >> totalCodeCnt >> embLen;
	
	InvertedIndexEntry *invertedIndex = new InvertedIndexEntry[totalCodeCnt];
	
	size_t tupleCnt = 0;
	for(int i = 0; i < totalCodeCnt; i++)
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

	sort(invertedIndex, invertedIndex + totalCodeCnt);
	
	code2Pos = new CodePos[totalCodeCnt];	
	invertedIndexValue = new GInfo[tupleCnt];	
	
	int pos = 0;
	for(int i = 0; i < totalCodeCnt; i++)
	{
		(code2Pos+i)->code = invertedIndex[i].code;
		(code2Pos+i)->pos = pos;
		for(int j = 0; j < invertedIndex[i].infos.size(); j++)
		{
			invertedIndexValue[pos] = invertedIndex[i].infos[j];
			pos++;
		}
	}

	cout << "Finish loading inverted index" << endl;
	cout << "Loading database done" << endl;
}

bool
Database::getCodeAndEmbByQidWithPython(const int qid, uint64_t &code, GInfo &qInfo)
{

	PyObject *ret = PyObject_CallFunction(getCodeAndEmbByQid, "i", qid);
	if (!ret || !PyTuple_Check(ret))
	{
		cout << "Failed to get code and embedding " << endl;
		PyRun_SimpleString("print('')");
		return false;
	} 	
	

	PyObject *py_code = PyTuple_GetItem(ret, 0);
	assert(PyLong_Check(py_code));
	code = PyLong_AsLong(py_code);
//	cout << code << endl;

	PyObject *py_emb = PyTuple_GetItem(ret, 1);
	assert(embLen = PyTuple_Size(py_emb));
	
	qInfo.gid = qid;
	//qInfo.code =  code;
	for(int i = 0; i < embLen; i++)
	{
		qInfo.emb[i] = PyFloat_AsDouble(PyTuple_GetItem(py_emb, i));
	}

	return true;
}

/**
 * Process Query identifeid by qid
 * @param qid: the graph id of query, should be loaded into data_fetcher in python
 * @param ret: into which the returned graph ids are inserted 
 */
bool
Database::QueryProcessGetCandidate(const int qid, const int ub, const int width,
			bool fineGrained,
			const graph &q, vector<int> &ret)
{
	/* get embedding and code */
	GInfo qInfo;
	uint64_t qCode;
 	bool retCode = getCodeAndEmbByQidWithPython(qid, qCode, qInfo);
	if (!retCode) return false;
	/* search all code within threshld */
	vector <uint64_t> validCode;	

	getAllValidCode(qCode, GED2Hamming[ub], totalCodeCnt, codeLen, code2Pos,
			validCode);
//	getAllValidCode(qCode, ub+1, totalCodeCnt, codeLen, code2Pos,
//			validCode);


	//cout << "get valid code done " << endl;
	/* get gids of graphs of those codes */
	int totalGraphCnt = graphDB.size();
	for(int i = 0; i < validCode.size(); i++)
	{
		int start = code2Pos[validCode[i]].pos;
		int end;
		if (validCode[i] == totalCodeCnt-1)
			end = totalGraphCnt;
		else
		 	end = code2Pos[validCode[i]+1].pos;
		for(int j = start; j < end; j++)
		{
			double dis = dist(qInfo, invertedIndexValue[j], 
						embLen);
			if (fineGrained > 0 && dis > (double)ub)
			{
				continue;
			}
			//uint64_t pos = gid2Pos[invertedIndexValue[j].gid];
//			candidateSet.push_back(graphDB[pos]);
			ret.push_back(invertedIndexValue[j].gid);
		}
	}

	return true;
}
/**
 * Process Query identifeid by qid
 * @param qid: the graph id of query, should be loaded into data_fetcher in python
 * @param ret: into which the returned graph ids are inserted 
 */
bool
Database::QueryProcess(const int qid, const int ub, const int width,
			bool fineGrained,
			const graph &q, vector<int> &ret, vector<int> &candGid)
{
	/* get embedding and code */
	GInfo qInfo;
	uint64_t qCode;

	struct timeval start, end;
	gettimeofday(&start, NULL);

 	bool retCode = getCodeAndEmbByQidWithPython(qid, qCode, qInfo);

	gettimeofday(&end, NULL);
	float timeuse = 1000000 * (end.tv_sec - start.tv_sec)
				 + end.tv_usec - start.tv_usec; 
	timeuse = timeuse * 1.0 / 1000000; 
	totalEncodeTime += timeuse;			 


	if (!retCode) return false;

	/* search all code within threshld */
	
	vector <uint64_t> validCode;	
	gettimeofday(&start, NULL);
	getAllValidCode(qCode, GED2Hamming[ub], totalCodeCnt, codeLen, code2Pos,
			validCode);
	//getAllValidCode(qCode, ub+1, totalCodeCnt, codeLen, code2Pos,
	//		validCode);


	/* get gids of graphs of those codes */
	//vector <graph> candidateSet;
	vector<int> candidateSet;
	int totalGraphCnt = graphDB.size();
	for(int i = 0; i < validCode.size(); i++)
	{
		int start = code2Pos[validCode[i]].pos;
		int end;
		if (validCode[i] == totalCodeCnt-1)
			end = totalGraphCnt;
		else
		 	end = code2Pos[validCode[i]+1].pos;
		for(int j = start; j < end; j++)
		{
			double dis = dist(qInfo, invertedIndexValue[j], 
						embLen);
//			if (fineGrained > 0 && dis > ((double)ub) + 0.5)
			if (fineGrained > 0 && dis > ((double)GED2Hamming[ub]-0.5))
			{
				continue;
			}
			uint64_t pos = gid2Pos[invertedIndexValue[j].gid];
//			assert(graphDB[pos].graph_id == invertedIndexValue[j].gid);
			//candidateSet.push_back(graphDB[pos]);
			candidateSet.push_back(pos);
			candGid.push_back(invertedIndexValue[j].gid);
		}
	}
	gettimeofday(&end, NULL);
	timeuse = 1000000 * (end.tv_sec - start.tv_sec)
				 + end.tv_usec - start.tv_usec; 
	timeuse = timeuse * 1.0 / 1000000; 
	totalSearchTime += timeuse;			 


	/* verification stage */
	gettimeofday(&start, NULL);
//	Verify(q, candidateSet, ub, width, ret);
	int i = 0, bound;
	for(; i < candidateSet.size();i++)
	{
		graph g = graphDB[candidateSet[i]];
		graph query = q; 
//		changeLabel(q.V, g.V);
		if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
		else bound = ub;
		BSEditDistance ed(width);
		int ged = ed.getEditDistance(query, g, bound);
		//cout << ged << endl;
		if (ged >= 0)
		{
			ret.push_back(g.graph_id);
		}
		FLAG = true;
	}

	gettimeofday(&end, NULL);
	timeuse = 1000000 * (end.tv_sec - start.tv_sec)
				 + end.tv_usec - start.tv_usec; 
	timeuse = timeuse * 1.0 / 1000000; 
	totalVerifyTime += timeuse;			 

	return true;
}

void changeLabel(vector<int> &v1, vector<int> &v2)
{
	map<int, int> labelMapper;
	int newLabel = 0;
	map<int, int>::iterator iter;
	for(int i = 0; i < v1.size(); i++)
	{
		iter = labelMapper.find(v1[i]);
		int y;
		if (iter == labelMapper.end())
		{
			labelMapper.insert(make_pair(v1[i], newLabel));
			y = newLabel;
			newLabel++;
			
		}
		else y = iter->second;
		v1[i] = y;
		
	}
	for(int i = 0; i < v2.size(); i++)
	{
		iter = labelMapper.find(v2[i]);
		int y;
		if (iter == labelMapper.end())
		{
			labelMapper.insert(make_pair(v2[i], newLabel));
			y = newLabel;
			newLabel++;
		}
		else y = iter->second;
		v2[i] = y;
	}

}

bool 
Database::Verify(const graph &query, const vector<graph> &candidates, 
		 const int ub, const int width,
		 vector<int> &ret)
{
	//vector <graph> queryDB; queryDB.push_back(query);
	int i = 0, bound;
	for(; i < candidates.size();i++)
	{
		graph g = candidates[i];
		graph q = query; 
//		changeLabel(q.V, g.V);
		if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
		else bound = ub;
		BSEditDistance ed(width);
		int ged = ed.getEditDistance(q, g, bound);
		//cout << ged << endl;
		if (ged >= 0)
		{
			ret.push_back(g.graph_id);
		}
		FLAG = true;
	}
	return true;
}

bool
Database::topKQueryProcess(const int qid, const int K, vector<uint64_t> &ret, int thres)
{
	/* get embedding and code */
	GInfo qInfo;
	uint64_t qCode;

 	bool retCode = getCodeAndEmbByQidWithPython(qid, qCode, qInfo);


	if (!retCode) return false;

	CompForTopK::embLen = embLen;
	CompForTopK::qInfo = qInfo;

	vector <uint64_t> validCode;
	deque <uint64_t> gids;
	int totalGraphCnt = graphDB.size();
	int retVal = getTopKByCode(K, qCode, codeLen, thres, 
				totalCodeCnt, totalGraphCnt, code2Pos, validCode);
	if (retVal > 0)
	{
		GInfo *candidates = new GInfo[retVal];
		int pos = 0;
		for(int i = 0; i < validCode.size(); i++)
		{
			int start = code2Pos[validCode[i]].pos;
			int end;
			if (validCode[i]+1 == totalCodeCnt)
				end = totalGraphCnt;
			else
				end = code2Pos[validCode[i]+1].pos;
			for(int j = start; j < end; j++)
			{
//				candidates[pos] = invertedIndexValue[j];
/*				if (invertedIndexValue[j].gid == 0)
				{
					cout << "invertedIndex gid = 0" << endl;
				}*/
				candidates[pos].gid = invertedIndexValue[j].gid;
				for(int dim = 0; dim < embLen; dim++)
					candidates[pos].emb[dim] = invertedIndexValue[j].emb[dim];
				pos++;
//				printf("%ld\n", invertedIndexValue[j].gid);
			}
		}
		assert(ret == pos);
		getTopKByEmb(K, retVal, candidates, gids);
		delete [] candidates;
	}
	else
	{
		getTopKByEmb(K, totalGraphCnt, invertedIndexValue, gids);
	}

	assert(gids.size() == K);
	for(int i = 0; i < K; i++)
	{
		ret.push_back(gids[i]);
	}

	return true;
}
