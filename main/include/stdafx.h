#ifndef _STDAFX_H
#define _STDAFX_H
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <stack>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <set>
#include <ctime>
#include <queue>
#include <functional>
#include <algorithm>
#include <random>
#include <list>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <bitset>
#include <stack>

#include <sys/time.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#define  _mkdir mkdir
#define  _access access
#define  ASSERT assert
#define _stati64 stat64
#define _fseeki64 fseeko
#define _FILE_OFFSET_BITS 64  

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

static int INSERTED = 253;
static int DELETED = 254;
static int UNMAPPED = 255;

typedef struct verifyGraphNode
{
	u8 verifyGraphNodeID;
	u8 verifyGraphNodeStr;
	verifyGraphNode(){}
	verifyGraphNode(const verifyGraphNode &t)
	{
		*this = t;
	}
	verifyGraphNode & operator= (const verifyGraphNode &t) //const 
	{
		if (this != &t)
		{
			this->verifyGraphNodeID = t.verifyGraphNodeID;
			this->verifyGraphNodeStr = t.verifyGraphNodeStr;
		}
		return *this;
	}
};

using namespace std;
#endif
