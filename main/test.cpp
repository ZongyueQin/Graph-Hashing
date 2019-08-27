//#include "python/Python.h"
#include<Python.h>
#include <iostream>
#include <string>
#include "BSED.h"
#include <sys/types.h>

using namespace std;

bool Verify(const graph &query, const vector<graph> &graphDB, 
		const int ub, const int width,
		vector<int> &ret)
{
	vector <graph> queryDB; queryDB.push_back(query);
	int i = 0, j, bound;
	for(; i < graphDB.size();i++)
	{
		graph g = graphDB[i];
		for(j = 0; j < queryDB.size(); j++)
		{
			graph q = queryDB[j]; 
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
	}
	return true;
}

int main(int argc, char **argv)
{
	Py_Initialize();
	// load module
	string path = ".";
	string chdir_cmd = string("sys.path.append(\"") + path + "\")";
	const char* cstr_cmd = chdir_cmd.c_str();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(cstr_cmd);

	PyObject *moduleName = PyUnicode_FromString("mytest");
	PyObject *pModule = PyImport_Import(moduleName);
	if (!pModule)
	{
		cout << "[ERROR] Python get module failed" << endl;
		return 0;
	}
	cout << "[INFO] Python get module succeed" << endl;

	//Load method
	PyObject *pv1 = PyObject_GetAttrString(pModule, "build");
	
	if (!pv1 || !PyCallable_Check(pv1))
	{
		cout << "[ERROR] Can't find function (build)" << endl;
		return 0;
	}	
	cout << "[INFO] Get function (build) succeed" << endl;

	PyObject *pv2 = PyObject_GetAttrString(pModule, "turn");
	
	if (!pv2 || !PyCallable_Check(pv2))
	{
		cout << "[ERROR] Can't find function (turn)" << endl;
		return 0;
	}	
	cout << "[INFO] Get function (turn) succeed" << endl;



	// set parameters
	PyObject* args = PyTuple_New(1);
	PyObject* arg1 = PyLong_FromLong(4);
	PyTuple_SetItem(args, 0, arg1);

	PyObject* myInt = PyObject_CallObject(pv1, args);
	PyObject* args2 = PyTuple_New(1);
	PyObject* arg2 = PyLong_FromLong(3);
	PyTuple_SetItem(args2, 0, arg2);
        PyObject* pRet = PyObject_CallObject(pv2, args2);

	if (pRet)
	{
		long result = PyLong_AsLong(pRet);
		cout << "result: " << result << endl;
	}

	if(argc < 7) {cout << "database n query m bound w" << endl; exit(0);}
	string db = argv[1]; 
	int totalGraph =  atoi(argv[2]);
	string query = argv[3];
	int totalQuery = atoi(argv[4]);
	const int ub = atoi(argv[5]);
	int width = atoi(argv[6]);
	const int sorted = 1;


	string db_out = db+"_ordered";
	graph::reOrderGraphs(db.c_str(), db_out.c_str(), totalGraph);
	vector<graph> graphDB = graph::readGraphMemory(db_out.c_str(), totalGraph);
	std::remove(db_out.c_str()); 

	int i = 0, j = 0, bound;	
	int sum = 0; 
	struct timeval start,end; 
	float timeuse; 
	gettimeofday(&start, NULL); 


	string query_out = query+"_ordered";
	graph::reOrderGraphs(query.c_str(), query_out.c_str(), totalQuery);
	vector<graph> queryDB = graph::readGraphMemory(query_out.c_str(), totalQuery);
	std::remove(query_out.c_str());
	i = 0;
	for(; i < graphDB.size();i++)
	{
		graph g = graphDB[i];
		for(j = 0; j < queryDB.size(); j++)
		{
			graph q = queryDB[j]; 
			if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
			else bound = ub;
			BSEditDistance ed(width);
			int ged = ed.getEditDistance(q, g, bound);
			//cout << ged << endl;
			FLAG = true;
		}
	}
	
	gettimeofday(&end, NULL); 
	timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec; 
	timeuse = timeuse * 1.0 / 1000000; 
	//cout << timeuse << endl;

	for(j = 0; j < queryDB.size(); j++)
	{
		vector<int> resultVec;
		Verify(queryDB[j], graphDB, ub, width, resultVec);
	}
	

	Py_Finalize();

	return 0;
	
	
}


