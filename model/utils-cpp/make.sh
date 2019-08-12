#!/bin/sh

g++ -std=c++11 -O3 processInvertedIndex.cpp -o processInvertedIndex

g++ -std=c++11 -O3 query.cpp -o query

g++ -std=c++11 -O2 topKQuery.cpp -o topKQuery

cp query ..
cp processInvertedIndex ..
cp topKQuery ..
