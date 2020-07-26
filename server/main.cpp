//#include "python/Python.h"
#include<Python.h>
#include <iostream>
#include <sstream>
#include <string>
//#include "BSED.h"
#include <sys/types.h>
#include "Database.h"
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <pthread.h>

using namespace std;

#define PORT 7000
#define CODELEN 32 // length of hash code
#define QUEUE 20 // length of connect request queue

const bool secondaryPruning = true;
const int width = 15;
Database *database;

int openListenFd(unsigned short port)
{
	int listenFd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenFd < 0)
	{
		cout << "Socket failed, exit" << endl;
		exit(-1);
	}

	struct sockaddr_in serverSockAddr;
	serverSockAddr.sin_family = AF_INET;
	serverSockAddr.sin_port = htons(port);
	serverSockAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	if(bind(listenFd, (struct sockaddr*) &serverSockAddr, sizeof(serverSockAddr)) == -1)
	{
		perror("bind");
		exit(-1);
	}

	if(listen(listenFd, QUEUE) == -1)
	{
		perror("listen");
		exit(-1);
	}

	return listenFd;

}

graph string2query(const string &str, int &ub, int &ansNumBound)
{
	//cout << "parsing..." << endl;
	graph q;
	istringstream is(str);
	int gid, v, e, f, t, l;
	is >> ub;
	//cout << ub << endl;
	is >> ansNumBound;
	//cout << ansNumBound << endl;
	is >> gid;
	q.graph_id = gid;

	is >> v >> e;
	q.v = v; q.e = e;
	q.V.resize(q.v, 0); vector<int> tmp(q.v, 255); q.E.resize(q.v, tmp);

	//cout << 'v' << endl;
	for(int i = 0; i < v; i++)
		is >> q.V[i];

	//cout << 'e' <<endl;
	for(int i = 0; i < e; i++)
	{
		is >> f >> t >> l;
		q.E[f][t] = l;
		q.E[t][f] = l;
	}
	return q;
}

void echo(int fd)
{
	char buf[1024];
	int len;
	memset(buf, 0, sizeof(buf));
	len = recv(fd, buf, sizeof(buf), 0);
	send(fd, buf, len, 0);
}

void query(int fd)
{
	char buf[1024];
	string queryStr = "";
	int len;
	vector<int> result;
	vector<int> candidates;

	int ub;
	graph q;
	int ansNumUpbound;



	while(1)
	{
		memset(buf, 0, sizeof(buf));
		len = recv(fd, buf, sizeof(buf), 0);
		queryStr = queryStr + string(buf);
		if (strcmp(buf+len-5, "done\n") == 0) break;

	}

	//cout << "queryStr: " << queryStr << endl;
	//Parse q by queryStr
	q = string2query(queryStr, ub, ansNumUpbound);

	bool retCode = database->QueryProcess(queryStr, ub, width,
					secondaryPruning,
					q, result, candidates);
	if (!retCode)
	{
		cout << "error when query " << q.graph_id << endl;
	}
	
	string ret = "";
	for(int i = 0; i < result.size(); i++)
	{
		if (i == ansNumUpbound) break;
		//send answer to client
		ret += to_string(result[i]);
		ret += ";";
	}

	send(fd, ret.c_str(), ret.length(), 0);

}

void *threadRoutine(void *vargp)
{
	int connfd = *((int *)vargp);
	int ret = pthread_detach(pthread_self());
	if (ret != 0)
	{
		perror("pthread_detach");
		return NULL;
	}
	free(vargp);
	query(connfd);
	close(connfd);
	return NULL;
}

int main(int argc, char **argv)
{
	if(argc < 6)
	{
		cout << "database n model_path " << 
			 "inv_idx_txt BitWeightFile, [mapper_file]" << endl; 
		exit(0);
	}

	int listenFd = openListenFd(PORT);
	socklen_t clientLen;
	int *connfdp;
	struct sockaddr_in clientAddr;


	string db_path = argv[1]; // the bss file where data graphs are stored.
	int totalGraph =  atoi(argv[2]); // how many graphs to read from @db_path

	string model_path = argv[3]; // .ckpt file to liad model

	string invIdxTxtPath = argv[4]; // .txt file to load inverted index

	string BitWeightsFile = string(argv[5]);

	string GED2HammingFile = "";
	if (argc == 7)
		GED2HammingFile = string(argv[6]);


	database = new Database(model_path, db_path, 
			invIdxTxtPath,
			totalGraph, CODELEN, 
			BitWeightsFile,
			GED2HammingFile);


	pthread_t tid;
	while(1)
	{
		clientLen = sizeof(clientAddr);
		connfdp = (int *)malloc(sizeof(int));
		*connfdp = accept(listenFd, (struct sockaddr*)&clientAddr, &clientLen);
		if(*connfdp < 0)
		{
			free(connfdp);
			continue;
		}
		else
		{
			int err = pthread_create(&tid, NULL, threadRoutine, connfdp);
			if (err != 0)
			{
				cout << "Failed pthread_create" << endl;
			}
		}
	}

	delete database;
	close(listenFd);
	return 0;
	
	
}


