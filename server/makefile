DIR_INC = ../main/include
DIR_SRC = ../main/src
DIR_OBJ = ./obj
DIR_BIN = ./bin

#cflags=-I/usr/include/python3.5m -I/usr/include/python3.5m  -Wno-unused-result -Wsign-compare -g -fstack-protector-strong -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes
#ldflags=-L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic -Wl,-O3 -Wl,-Bsymbolic-functions
cflags=-I/usr/include/python3.6m -I/usr/include/python3.6m  -Wno-unused-result -Wsign-compare -g -fdebug-prefix-map=/build/python3.6-0aiVHW/python3.6-3.6.9=. -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O3 -Wall
ldflags=-L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/usr/lib -lpython3.6m -lpthread -ldl  -lutil -lm  -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -fPIC


SRC = $(wildcard ${DIR_SRC}/*.cpp)  
OBJ = $(patsubst %.cpp,${DIR_OBJ}/%.o,$(notdir ${SRC})) 

TARGET = main

BIN_TARGET = ${DIR_BIN}/${TARGET}

CC = g++
CFLAGS = -std=c++0x -g -Wall -I${DIR_INC} -L$(DIR_OBJ) -D_FILE_OFFSET_BITS=64 -D_LARGE_FILE 

${BIN_TARGET}:${OBJ} main.cpp
	$(CC) $(OBJ) main.cpp $(ldflags) $(cflags) $(CFLAGS) -o $@     

${DIR_BIN}/topK:${OBJ} topK.cpp
	$(CC) $(OBJ) topK.cpp $(ldflags) $(cflags) $(CFLAGS) -o $@     


${DIR_BIN}/test:${OBJ} test.cpp
	$(CC) $(OBJ) test.cpp $(ldflags) $(cflags) $(CFLAGS) -o $@     


${DIR_OBJ}/%.o:${DIR_SRC}/%.cpp
	$(CC) $(CFLAGS) -c  $<  $(ldflags) $(cflags) -o $@

clean:
	rm ${DIR_OBJ}/*.o bin/main bin/test
