#!/bin/sh

g++ -std=c++11 -O3 processInvertedIndex.cpp -o processInvertedIndex

g++ -std=c++11 -O3 query.cpp -o query

cp query ..
cp processInvertedIndex ..
