#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "BSED.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
static graph getGraphFromCString(const char *str)
{
	stringstream ss(str);
//	sscanf(str, "%d", &gid);
//	sscanf(str, "%d %d", &v, &e);
	int gid, v, e;
	int f, t, l;
	ss >> gid >> v >> e;
	graph g;
	g.graph_id = gid; g.v = v; g.e = e;
	g.V.resize(g.v, 0); vector<int> tmp(g.v, 255); g.E.resize(g.v, tmp);

	for (int i = 0; i < v; i++)
		ss >> g.V[i]; //sscanf(fr, "%d\n", &g.V[i]);

	for (int i = 0; i < e; i++)
	{
		ss >> f >> t >> l; // sscanf(fr, "%d %d %d\n", &f, &t, &l);
		g.E[f][t] = l;
		g.E[t][f] = l;
	}
	return g;
}

static PyObject* getGED(PyObject *self, PyObject *args)
{
	const char *g1_string;
	const char *g2_string;
	int ub, width, bound;
	int ok;
	// get arguments ub,width, get two graphs g and q
	ok = PyArg_ParseTuple(args, "iiss", &ub, &width, &g1_string, &g2_string);	
	if (!ok)
	{
		return NULL;
	}
	graph g1 = getGraphFromCString(g1_string);
	graph g2 = getGraphFromCString(g2_string);
 
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


