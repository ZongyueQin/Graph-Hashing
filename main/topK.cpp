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
	if(argc < 8)
	{
		cout << "database n query m  K model_path " << 
			 "inv_idx_txt [mapper_file]" << endl; 
		exit(0);
	}
	string db_path = argv[1]; 
	int totalGraph =  atoi(argv[2]);
	string query = argv[3];
	int totalQuery = atoi(argv[4]);
//	const int ub = atoi(argv[5]);
	int K = atoi(argv[5]);
	string model_path = argv[6];

	string invIdxTxtPath = argv[7];

	string GED2HammingFile = "";
	if (argc == 9)
		GED2HammingFile = string(argv[8]);

	Database database(model_path, db_path, 
			invIdxTxtPath,
			totalGraph, CODELEN, GED2HammingFile);
	
	string query_out = query+"_ordered";
	graph::reOrderGraphs(query.c_str(), query_out.c_str(), totalQuery);
	vector<graph> queryDB = graph::readGraphMemory(query_out.c_str(), 
							totalQuery);
	std::remove(query_out.c_str());


	vector<uint64_t> result;

	stringstream ss;
	ss << "output/AIDS_k0_top" << K << "_" << OUTPUTFILE;
	string outputFile = ss.str();
	ofstream fout(outputFile);

	struct timeval start,end; 
	float timeuse, totalTime = 0; 

	for(int i = 0; i < queryDB.size();i++)
	{
		result.clear();
		graph q = queryDB[i];
		gettimeofday(&start, NULL); 
		bool retCode = database.topKQueryProcess(queryDB[i].graph_id, K, result);
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
	
	cout << "average response time = " << 
		totalTime/queryDB.size() << " s" << endl;
		

	return 0;
	
	
}


