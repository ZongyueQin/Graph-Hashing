#include "BSED.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT 12345
#define BUFSIZE 20000
void changeLabel(vector<int> &v1, vector<int> &v2)
{
	map<int, int> labelMapper;
	int newLabel = 0;
	map<int, int>::iterator iter;
	for(int i = 0; i < v1.size(); i++)
	{
		iter = labelMapper.find(v1[i]);
		int y;
		if (iter == labelMapper.end())
		{
			labelMapper.insert(make_pair(v1[i], newLabel));
			y = newLabel;
			newLabel++;
			
		}
		else y = iter->second;
		v1[i] = y;
		
	}
	for(int i = 0; i < v2.size(); i++)
	{
		iter = labelMapper.find(v2[i]);
		int y;
		if (iter == labelMapper.end())
		{
			labelMapper.insert(make_pair(v2[i], newLabel));
			y = newLabel;
			newLabel++;
		}
		else y = iter->second;
		v2[i] = y;
	}
	labelMapper.clear();

}
int main(int argc, char *argv[])
{
	/* set up server */	
	/*
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
	*/

	/* load database */
	if(argc < 7) {cout << "database n query m bound w" << endl; exit(0);}
	string db = argv[1]; 
	int totalGraph =  atoi(argv[2]);
	string query = argv[3];
	int totalQuery = atoi(argv[4]);
	const int ub = atoi(argv[5]);
	int width = atoi(argv[6]);
	const int sorted = 1;

//	char *db_out = sorted ? "./ged_db" : db.c_str(); 
//	char *query_out = sorted ? "./ged_query" : query.c_str();

	string db_out = db+"_ordered";
//	if(sorted){
	graph::reOrderGraphs(db.c_str(), db_out.c_str(), totalGraph);
//			cout << "preprocessing graph order done" << endl;
//	}
	
	vector<graph> graphDB = graph::readGraphMemory(db_out.c_str(), totalGraph);
	std::remove(db_out.c_str()); 
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
		for(j = 0; j < queryDB.size(); j++)
		{
			graph g = graphDB[i];
			graph q = queryDB[j]; 
			if (q.graph_id == g.graph_id)
			{
				cout << 0 << endl;
			}
			else
			{
			//	if(ub == -1)  bound = max(g.v, q.v) + g.e + q.e;
			//	else bound = ub;
				bound = max(g.v,q.v)+g.e+q.e;
				if (ub != -1 && ub < bound) bound = ub;
				BSEditDistance ed(width);
				changeLabel(g.V, q.V);
	//		cout << g.graph_id << ' ' << q.graph_id << endl;
				int ged = ed.getEditDistance(q, g, bound);
//			if(ged > -1) 
//			{
//				cout << g.graph_id << " " << q.graph_id << " " << ged << endl; 
//				sum++;
//			}
				cout << ged << endl;
				FLAG = true;
			}
		}
	}
	
//	cout << "total expand node " << totalExpandNode << endl;
//	cout << "initFilter and total results " << initFilter  << " " << sum << endl;
	gettimeofday(&end, NULL); 
	timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec; 
	timeuse = timeuse * 1.0 / 1000000; 
	cout << timeuse << endl;
	//}
	return 0;
}
