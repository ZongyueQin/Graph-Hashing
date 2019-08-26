#include "BSED.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <map>
#include <sstream>

#define PORT 12345
#define BUFSIZE 20000

int main(int argc, char *argv[])
{
	/* set up server */	
	char buf[BUFSIZE];
	int sockfd, len;

	struct sockaddr_in serv_addr, cli_addr;
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr=INADDR_ANY;
	serv_addr.sin_port=htons(PORT);

	if ((sockfd=socket(PF_INET, SOCK_DGRAM,0))<0)
	{
		cout << "Error creating socket" << endl;
		exit(0);
	}

	if(bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(struct sockaddr))<0)
	{
		cout << "bind error" << endl;
		exit(0);
	}

	unsigned sin_size = sizeof(struct sockaddr_in);
	

	/* load database */
	if(argc < 6) {cout << "database n query m w" << endl; exit(0);}
	string db = argv[1]; 
	int totalGraph =  atoi(argv[2]);
	string query = argv[3];
	int totalQuery = atoi(argv[4]);
//	const int ub = atoi(argv[3]);
	int width = atoi(argv[5]);
	const int sorted = 1;

//	char *db_out = sorted ? "./ged_db" : db.c_str(); 
//	char *query_out = sorted ? "./ged_query" : query.c_str();

	string db_out = db+"_ordered";
//	if(sorted){
//	graph::reOrderGraphs(db.c_str(), db_out.c_str(), totalGraph);
	
	vector<graph> graphDB = graph::readGraphMemory(db.c_str(), totalGraph);
	std::remove(db_out.c_str()); 
//		cout << "open query file" << query << endl;
	string query_out = query+"_ordered";
//	graph::reOrderGraphs(query.c_str(), query_out.c_str(), totalQuery);
	vector<graph> queryDB = graph::readGraphMemory(query.c_str(), totalQuery);

	int result[800][800];
	cout << "start" << endl;
	for(int i = 0; i < graphDB.size(); i++)
	{
		graph g = graphDB[i];
		for(int j = 0; j < queryDB.size(); j++)
		{
			cout << i << ' ' << j << endl;
			graph q = queryDB[j];
				int bound = 1;
//				if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
//				else bound = ub;
				BSEditDistance ed(width);
				int ged = ed.getEditDistance(q, g, 1);
 				result[i][j] = ged;
		}
	}	
	cout << "end" << endl;

	std::remove(query_out.c_str());
	map <int, int> gid2idx_g;
	map <int, int> gid2idx_q;
	for(int i = 0; i < graphDB.size(); i++)
		gid2idx_g.insert(make_pair(graphDB[i].graph_id, i));
	for(int i = 0; i < queryDB.size(); i++)
		gid2idx_q.insert(make_pair(queryDB[i].graph_id, i));
	/*
	while(true)
	{
		len = recvfrom(sockfd, buf, BUFSIZE, 0,(struct sockaddr*)&cli_addr, &sin_size);
		if (len < 0)
		{
			perror ("Error receiving data");
		}
		len = sendto(sockfd, buf, len, 0, (struct sockaddr *)&cli_addr, sin_size);
		if (len < 0)
		{
			perror("Error sending data");
		}
	}

	*/
	int i = 0, j = 0;	
	int sum = 0; 
	struct timeval start,end; 
	float timeuse; 
	//gettimeofday(&start, NULL); 

	cout << "Ready" << endl;
	/* wait for query */
	while (true)
	{
		memset(buf, 0, sizeof(buf));
		len = recvfrom(sockfd, buf, BUFSIZE, 0, (struct sockaddr*)&cli_addr, &sin_size);
		if (len < 0)
		{
			perror("Error receiving data");
		}
		int qid = atoi(buf);
		int qidx = gid2idx_q[qid];
//		cout << "server: " << queryDB[0].v << ' ' << queryDB[0].e << endl;

		memset(buf, 0, sizeof(buf));
		len = recvfrom(sockfd, buf, BUFSIZE, 0, (struct sockaddr*)&cli_addr, &sin_size);
		if (len < 0)
		{
			perror("Error receiving data");
		}
		const int ub = 1;//atoi(buf);
		//cout << "ub=" << ub << endl;
		stringstream ret;

		bool end = false;
		while (true)
		{
			memset(buf, 0, sizeof(buf));
			len = recvfrom(sockfd, buf, BUFSIZE, 0, (struct sockaddr*)&cli_addr, &sin_size);
//			cout << "receive inputs" << endl;
			if (len < 0)
			{
				perror("Error receiving data");
			}
			string input_string = string(buf);
			stringstream ss(input_string);
			int gid;
			for(i = 0; i < 1000; i++)
			{
				ss >> gid;
				//cout << "gid=" << gid << endl;
				if (gid < 0)
				{
					end = true;
					break;
				}
				int idx = gid2idx_g[gid];
				graph g = graphDB[idx];
				assert(g.graph_id == gid);
				graph q = queryDB[qidx];
				int bound;
				if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
				else bound = ub;
				BSEditDistance ed(width);
				int ged = ed.getEditDistance(q, g, bound);
				ret << ged << ' ';
				if (ged != result[idx][qidx]) cout << "result mismatch" << endl;
			}
			if (end) break;
		}

		len = sendto(sockfd, ret.str().c_str(), ret.str().length(), 0, (struct sockaddr *)&cli_addr, sin_size);
		if (len < 0)
		{
			perror("Error sending data");
		}

	}
	
//	cout << "total expand node " << totalExpandNode << endl;
//	cout << "initFilter and total results " << initFilter  << " " << sum << endl;
//	gettimeofday(&end, NULL); 
//	timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec; 
//	timeuse = timeuse * 1.0 / 1000000; 
	return 0;
}
