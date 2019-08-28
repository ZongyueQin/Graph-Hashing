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

// return is in res, the index (position) in Code2Pos
void getAllValidCode(uint64_t code, int thres, 
				int totalCodeCnt, int codeLen, 
				CodePos *index, vector<uint64_t> &res)
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


/**
 Construtor
 @param model_path: the path to load model
 @db: the path to load database graphs
 @graphNum: how many graphs to load
 */
Database::Database(const string &modelPath, const string &db, 
			const string &invIdxIdxPath,
			const string &invIdxValPath,
			int totalGraph, int _embLen, int _codeLen,
			int _totalCodeCnt)
{
	embLen = _embLen;
	codeLen = _codeLen;
	totalCodeCnt = _totalCodeCnt;
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
	string db_out = db+"_ordered";
	graph::reOrderGraphs(db.c_str(), db_out.c_str(), totalGraph);
	graphDB = graph::readGraphMemory(db_out.c_str(), totalGraph);
	std::remove(db_out.c_str()); 
	for(int i = 0; i < graphDB.size(); i++)
		gid2Pos.insert(make_pair(graphDB[i].graph_id, i));

	/* load inverted index file */
	int fd1 = open(invIdxIdxPath.c_str(), O_RDWR, 00777);
	code2Pos = (CodePos*) mmap(NULL, 2*sizeof(uint64_t)*totalCodeCnt, 
					PROT_READ, 
					MAP_SHARED,
					fd1, 0);
	if (code2Pos == (void *)-1)
	{
		fprintf(stderr, "mmap: %s\n", strerror(errno));
		exit(0);
	}

	close(fd1);

	int fd2 = open(invIdxValPath.c_str(), O_RDWR, 00777);
	invertedIndexValue = (GInfo*) mmap(NULL, totalGraph*sizeof(GInfo),
						PROT_WRITE,
						MAP_PRIVATE|MAP_LOCKED,
						fd2, 0);
	if (invertedIndexValue == (void*)-1)
	{
		fprintf(stderr, "mmap2: %s\n", strerror(errno));
		exit(0);
	}
	close(fd2);


}

bool
Database::getCodeAndEmbByQidWithPython(const int qid, uint64_t &code, GInfo &qInfo)
{
//	cout << "DATABASE: " << qid << endl;
	/*py_qid = PyLong_FromLong(qid);
	if (py_qid == NULL)
	{
		cout << "Failed FromLong" << endl;
		return false;
	}
	arg = PyTuple_New(1);
	if (arg == NULL)
	{
		cout << "Failed PyTuple_New" << endl;
		return false;
	}
	int err = PyTuple_SetItem(arg, 0, py_qid);
	if (err)
	{
		cout << "Failed SetItem" << endl;
		return false;
	}
	ret = PyObject_CallObject(getCodeAndEmbByQid, arg);
	*/
	PyObject *ret = PyObject_CallFunction(getCodeAndEmbByQid, "i", qid);
	if (!ret || !PyTuple_Check(ret))
//	if (!ret)
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
		//py_dim = PyTuple_GetItem(py_emb, i);
		//double dim = PyFloat_AsDouble(py_dim);
		//qInfo.emb[i] = dim;
		qInfo.emb[i] = PyFloat_AsDouble(PyTuple_GetItem(py_emb, i));
//		cout << qInfo.emb[i] << ' ';
	}
//	cout << endl;
	
//	Py_DECREF(py_qid);
//	Py_DECREF(arg);
//	Py_DECREF(ret);
//	Py_DECREF(py_code);
//	Py_DECREF(py_emb);
	
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
			const graph &q, vector<int> &ret)
{
	/* get embedding and code */
	//cout << "encode " << qid << endl;
	GInfo qInfo;
	uint64_t qCode;
 	bool retCode = getCodeAndEmbByQidWithPython(qid, qCode, qInfo);
	if (!retCode) return false;

	/* search all code within threshld */
	//cout << "search " << endl;
	vector <uint64_t> validCode;	

	getAllValidCode(qCode, ub+1, totalCodeCnt, codeLen, code2Pos,
			validCode);

	//cout << "get valid code done " << endl;
	/* get gids of graphs of those codes */
	vector <graph> candidateSet;
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
			uint64_t pos = gid2Pos[invertedIndexValue[j].gid];
			candidateSet.push_back(graphDB[pos]);
		}
	}

	//cout << "verify" << endl;
	/* verification stage */
	Verify(q, candidateSet, ub, width, ret);
	return true;
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


