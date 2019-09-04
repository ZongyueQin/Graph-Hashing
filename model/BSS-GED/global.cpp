#include "global.h"

extern int max_v_1 = 0;
extern int max_e_1 = 0;
extern int max_v_2 = 0;
extern int max_e_2 = 0;
extern int max_d_1 = 0;
extern int max_d_2 = 0;

extern vector<vector<int > > adjList1(0, vector<int>());
extern vector<vector<int > > adjList2(0, vector<int>());
extern verifyGraphNode gn1[512] = { verifyGraphNode() };
extern verifyGraphNode gn2[512] = { verifyGraphNode() };
extern u8 a1[512][512] = {0}; //
extern u8 a2[512][512] = {0};
extern vector<int> vs1(0, 0);
extern vector<int> vs2(0, 0);
extern vector<int> group1(0, 0);
extern vector<int> group2(0, 0);

extern u8 start_deleted[64] = {0};
extern u8 end_deleted[64] = {0};
extern u8 tmpDegree1[512] ={0};
extern u8 tmpDegree2[512] = {0};
extern u8 edgeList1[64] = {0};
extern u8 edgeList2[64] = {0};
extern u8 edge_set_1[512] = {0};
extern u8 edge_set_2[512] = {0};
extern u8 succ_degree_1[512] = {0};
extern u8 succ_degree_2[512] = {0};
extern int ds1[512] = {0};
extern int ds2[512] = {0};

extern bool groupFlag1[512] = {0};
extern bool groupFlag2[512] = {0};
extern int groupNumber = 0;
extern bool unionFlag1[512] = {0};
extern bool unionFlag2[512] = {0};

extern bool FLAG = true;
extern bool VERTEXFLAG1 = false;
extern bool VERTEXFLAG2 = false;


extern u64 total_1 = 0;
extern u64 total_2 = 0;
extern u64 totalExpandNode = 0;
extern u64 total_search_node = 0;
