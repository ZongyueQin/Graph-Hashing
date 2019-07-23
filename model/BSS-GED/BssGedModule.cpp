#include <Python.h>
#include "BSED.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
static void getGraphFromCString(const char *str, graph &g)
{
	std::stringstream ss;
	ss << str;
	cout << ss.str() << endl;
	int gid, v, e;
	int f, t, l;
	ss >> gid >> v >> e;
//	sscanf(str, "%d", &gid);
//	sscanf(str, "%d %d", &v, &e);
	
	g.graph_id = gid; g.v = v; g.e = e;
	g.V.resize(g.v, 0); vector<int> tmp(g.v, 255); g.E.resize(g.v, tmp);

	for (int i = 0; i < v; i++)
		ss >> g.V[i]; 
//		sscanf(str, "%d", &g.V[i]);

	for (int i = 0; i < e; i++)
	{
		ss >> f >> t >> l; 
		//sscanf(str, "%d %d %d", &f, &t, &l);
		g.E[f][t] = l;
		g.E[t][f] = l;
	}
//	return g;
}

static PyObject* getGED(PyObject *self, PyObject *args)
{
//	const char *g1_string;
//	const char *g2_string;
	const char *filename_1;
	const char *filename_2;
	int ub, width, bound;
	int ok;
	// get arguments ub,width, get two graphs g and q
	ok = PyArg_ParseTuple(args, "iiss", &ub, &width, &filename_1, &filename_2);	
	if (!ok)
	{
		return NULL;
	}
//	graph *g1 = new graph;
//	graph *g2 = new graph;
//	getGraphFromCString(g1_string, *g1)
	string string_1 = string(filename_1) + "_ordered";
	string string_2 = string(filename_2) + "_ordered";
	graph::reOrderGraphs(filename_1, string_1.c_str(), 1);
	graph::reOrderGraphs(filename_2, string_2.c_str(), 1);
	graph g1 = graph::readGraphMemory(string_1.c_str(), 1)[0];
	graph g2 = graph::readGraphMemory(string_2.c_str(), 1)[0];
	
//	getGraphFromCString(g2_string, *g2);
	 
	ub=-1;
	if(ub == -1)  bound = max(g1.v, g2.v) + g1.e + g2.e;
	else bound = ub;
	BSEditDistance ed(width);
	int ged = ed.getEditDistance(g1, g2, bound);
	return PyLong_FromLong(ged);
}

static PyMethodDef BssMethods[]={
	{"getGED", getGED, METH_VARARGS, "Get GED between 2 graphs."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef BssGedModule={
	PyModuleDef_HEAD_INIT,
	"BssGed",
	NULL,
	-1,
	BssMethods
};

extern "C"
PyMODINIT_FUNC
PyInit_BssGed(void)
{
	return PyModule_Create(&BssGedModule);
}


