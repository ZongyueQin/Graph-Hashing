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

	vector <graph> graphDB;
	const string pythonPath = "../model";
	PyObject *getCodeAndEmbByQid;
	CodePos *code2Pos;
	GInfo *invertedIndexValue;
	map <uint64_t, uint64_t> gid2Pos;

//	PyObject *py_qid, *arg, *ret, *py_code, *py_emb, *py_dim;
public:
	Database(const string &modelPath, const string &db,  
		 const string &invIdxIdxPath, 
		 const string &invIdxValPath,
		 int totalGraph, int _embLen, int _codeLen,
		 int _totalCodeCnt);
	~Database()
	{
		int totalGraphCnt = graphDB.size();
		munmap((void*)code2Pos, 2*sizeof(uint64_t)*totalCodeCnt);
		munmap((void*)invertedIndexValue, sizeof(GInfo)*totalGraphCnt);
		Py_Finalize();
	}

	bool QueryProcess(const int qid, const int ub, const int width,
		bool fineGrained, const graph &q, vector<int> &ret);
private:
	bool Verify(const graph &query, const vector<graph> &candidates, 
		 const int ub, const int width,
		 vector<int> &ret);

	bool getCodeAndEmbByQidWithPython(const int qid, uint64_t &code, GInfo &qInfo);


};
