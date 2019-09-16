target: bitmap.o graph.o verifyGraph.o global.o treeNode.o BSED.o ged
CC=g++
CFLAGS= -O3 -std=c++0x -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE 

ged: bitmap.o graph.o verifyGraph.o global.o treeNode.o BSED.o ged.cpp 
	$(CC) $(CFLAGS) bitmap.o graph.o verifyGraph.o global.o treeNode.o BSED.o ged.cpp -o ged
bitmap.o: bitmap.cpp
	$(CC) -c $(CFLAGS) bitmap.cpp
graph.o: graph.cpp
	$(CC) -c $(CFLAGS) graph.cpp
verifyGraph.o: verifyGraph.cpp
	$(CC) -c $(CFLAGS) verifyGraph.cpp
global.o: global.cpp
	$(CC) -c $(CFLAGS) global.cpp
treeNode.o: treeNode.cpp
	$(CC) -c $(CFLAGS) treeNode.cpp 
BSED.o: BSED.cpp
	$(CC) -c $(CFLAGS) BSED.cpp
clean:
	rm -rf main *.o *~ ged 

