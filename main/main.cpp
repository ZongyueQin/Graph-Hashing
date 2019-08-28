//#include "python/Python.h"
#include<Python.h>
#include <iostream>
#include <sstream>
#include <string>
//#include "BSED.h"
#include <sys/types.h>
#include "Database.h"

using namespace std;

#define CODELEN 32
#define OUTPUTFILE "output.txt"

int main(int argc, char **argv)
{
	if(argc < 10)
	{
		cout << "database n query m  w model_path " << 
			 "inv_idx_txt inv_idx_idx inv_idx_val " << endl; 
		exit(0);
	}
	string db_path = argv[1]; 
	int totalGraph =  atoi(argv[2]);
	string query = argv[3];
	int totalQuery = atoi(argv[4]);
//	const int ub = atoi(argv[5]);
	int width = atoi(argv[5]);
	string model_path = argv[6];

	string invIdxTxtPath = argv[7];
	ifstream fin(invIdxTxtPath);
	if (!fin)
	{
		cout << "Unable to open " << invIdxTxtPath << endl;
		return 0;
	}
	int totalCodeCnt, embLen;
	fin >> totalCodeCnt >> embLen;
	fin.close();

	string invIdxIdxPath = argv[8];
	string invIdxValPath = argv[9];

	Database database(model_path, db_path, 
			invIdxIdxPath, invIdxValPath,
			totalGraph, embLen, CODELEN, totalCodeCnt);
	
	string query_out = query+"_ordered";
	graph::reOrderGraphs(query.c_str(), query_out.c_str(), totalQuery);
	vector<graph> queryDB = graph::readGraphMemory(query_out.c_str(), 
							totalQuery);
	std::remove(query_out.c_str());


	vector<int> result;

	for(int ub = 1; ub < 10; ub++)
	{
		//database.QueryProcess(queryDB[0].graph_id, result);
		stringstream ss;
		ss << "t=" << ub << "-" << OUTPUTFILE;
		string outputFile = ss.str();
		ofstream fout(outputFile);

		struct timeval start,end; 
		float timeuse, totalTime = 0; 

		bool useFineGrainWhenPrune = true, retCode;
		for(int i = 0; i < queryDB.size();i++)
		{
			graph q = queryDB[i];
			gettimeofday(&start, NULL); 
			retCode = database.QueryProcess(queryDB[i].graph_id, ub, width,
					useFineGrainWhenPrune,
					q, result);
			if (!retCode)
			{
				cout << "error when query " << q.graph_id << endl;
				break;
			}
			gettimeofday(&end, NULL); 
			timeuse = 1000000 * (end.tv_sec - start.tv_sec)
				 + end.tv_usec - start.tv_usec; 
			timeuse = timeuse * 1.0 / 1000000; 
			totalTime += timeuse;			 

			fout << q.graph_id << ' ';
			for(int j = 0; j < result.size(); j++)
			{
				fout << result[j] << ' ';
			}
			fout << endl;
		}
	
		cout << "ub = " << ub << ", average response time = " << 
			totalTime/queryDB.size() << " s" << endl;
	}


	return 0;
	
	
}


