#include <Python.h>
#include <iostream>
#include "BSED.h"
#ifndef UTILS
#include "utils.h"
#endif
#include <sys/types.h>
#include <string>
#include <map>

class Database
{
private:
	int embLen, codeLen;
	int totalCodeCnt;
	double *bit_weights;

	vector <graph> graphDB;
	const string pythonPath = "../model";
	PyObject *getCodeAndEmbByQid;
	CodePos *code2Pos;
	GInfo *invertedIndexValue;
	map <uint64_t, uint64_t> gid2Pos;

public:
	float totalEncodeTime, totalSearchTime, totalVerifyTime;
	int GED2Hamming[10];

public:
	Database(const string &modelPath, const string &db,  
		 const string &invIndex,
		 int totalGraph, int _codeLen,
		 const string &BitWeightsFile,
		 const string &GED2HammingFile="");
	~Database()
	{
		int totalGraphCnt = graphDB.size();
		munmap((void*)code2Pos, 2*sizeof(uint64_t)*totalCodeCnt);
		munmap((void*)invertedIndexValue, sizeof(GInfo)*totalGraphCnt);
		Py_Finalize();
	}
	bool QueryProcessGetCandidate(const int qid, const int ub, const int width,
		bool fineGrained, const graph &q, vector<int> &ret);

	bool QueryProcess(const int qid, const int ub, const int width,
		bool fineGrained, const graph &q, vector<int> &ret, vector<int> &candGid);

	bool topKQueryProcess(const int gid, const int K, vector<uint64_t> &ret, int thres = 11); 
private:
	bool Verify(const graph &query, const vector<graph> &candidates, 
		 const int ub, const int width,
		 vector<int> &ret);

	bool getCodeAndEmbByQidWithPython(const int qid, uint64_t &code, GInfo &qInfo);


};
