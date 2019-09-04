#ifndef _GLOBAL_H
#define _GLOBAL_H
#include "bitmap.h"
#include "verifyGraph.h"

extern int max_v_1;
extern int max_e_1;
extern int max_v_2;
extern int max_e_2;
extern int max_d_1;
extern int max_d_2;

extern vector<vector<int > > adjList1;
extern vector<vector<int > > adjList2;
extern verifyGraphNode gn1[512];
extern verifyGraphNode gn2[512];
extern u8 a1[512][512]; //
extern u8 a2[512][512];
extern vector<int> vs1;
extern vector<int> vs2;
extern vector<int> group1;
extern vector<int> group2;

extern u8 start_deleted[64];
extern u8 end_deleted[64];
extern u8 tmpDegree1[512];
extern u8 tmpDegree2[512];
extern u8 edgeList1[64];
extern u8 edgeList2[64];
extern u8 edge_set_1[512];
extern u8 edge_set_2[512];
extern u8 succ_degree_1[512]; 
extern u8 succ_degree_2[512];

extern int ds1[512];
extern int ds2[512];
extern bool FLAG;
extern bool VERTEXFLAG1;
extern bool VERTEXFLAG2;
extern bool groupFlag1[512];
extern bool groupFlag2[512];
extern int groupNumber;
extern bool unionFlag1[512];
extern bool unionFlag2[512];


extern u64 total_1;
extern u64 total_2;
extern u64 totalExpandNode; 
extern u64 total_search_node;
#endif
