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
#define dataset "0625_NoHash_FULL_ALCHEMY"
#define PRUNE
bool secondaryPruning = true;

int main(int argc, char **argv)
{
	if(argc < 9)
	{
		cout << "database n query m  w model_path " << 
			 "inv_idx_txt BitWeightFile, [mapper_file]" << endl; 
		exit(0);
	}
	string db_path = argv[1]; // the bss file where data graphs are stored.
	int totalGraph =  atoi(argv[2]); // how many graphs to read from @db_path
	string query = argv[3]; // the bss file where query graphs are stored
	int totalQuery = atoi(argv[4]); // how many queries to read from @query
//	const int ub = atoi(argv[5]);
	int width = atoi(argv[5]); // the beam width used for BSS-GED, 
                                   // recommend 15
	string model_path = argv[6]; // .ckpt file to liad model

	string invIdxTxtPath = argv[7]; // .txt file to load inverted index

	string BitWeightsFile = string(argv[8]);

	string GED2HammingFile = "";
	if (argc == 10)
		GED2HammingFile = string(argv[9]);

	Database database(model_path, db_path, 
			invIdxTxtPath,
			totalGraph, CODELEN, 
			BitWeightsFile,
			GED2HammingFile);
	
	string query_out = query+"_ordered";
        // Note BSS-GED need to call reOrderGraphs before call readGraphMemory
	graph::reOrderGraphs(query.c_str(), query_out.c_str(), totalQuery);
	vector<graph> queryDB = graph::readGraphMemory(query_out.c_str(), 
							totalQuery);
	std::remove(query_out.c_str());


	vector<int> result;
	vector<int> candidates;

	for(int ub = 0; ub < 7; ub++)
	{
		stringstream ss;
		ss << "output/" << dataset << "_t=" << ub << "_" << OUTPUTFILE;
		string outputFile = ss.str();
		ofstream fout(outputFile);

		stringstream ss2;
		ss2 << "output/" << dataset << "_t=" << ub << "_" << "candidate.txt";
		string candidateFile = ss2.str();
		ofstream fout2(candidateFile);

		struct timeval start,end; 
		float timeuse, totalTime = 0; 

		bool retCode;
		for(int i = 0; i < queryDB.size();i++)
		{
			result.clear();
			candidates.clear();
			graph q = queryDB[i];
			gettimeofday(&start, NULL); 
#ifdef PRUNE
			retCode = database.QueryProcess(queryDB[i].graph_id, ub, width,
					secondaryPruning,
					q, result, candidates);
#else
			retCode = database.directVerify(queryDB[i].graph_id, ub, width,
					q, result);
#endif
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

			fout2 << q.graph_id << ' ';
			for(int j = 0; j < candidates.size(); j++)
			{
				fout2 << candidates[j] << ' ';
			}
			fout2 << endl;

		}
	
		cout << "ub = " << ub << ", average response time = " << 
			totalTime/queryDB.size() << " s" << endl;
#ifdef PRUNE
		cout << "average encode time: " << database.totalEncodeTime / queryDB.size() << "; ";
		cout << "average search time: " << database.totalSearchTime / queryDB.size() << "; ";
		cout << "average verify time: " << database.totalVerifyTime / queryDB.size() << "; ";
#endif
		cout << "results written in " << ss.str() << endl;
		database.totalEncodeTime = 0;
		database.totalSearchTime = 0;
		database.totalVerifyTime = 0;
	}


	return 0;
	
	
}


