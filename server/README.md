# Query String Format

First line: QT,MAN,ID (QT: Query Threshold, MAN: Maximum Returned Answer Number, ID: query ID) 

Second line: n 

Third line: m 

(n: number of vertices, m, number of edges)

Next n lines, each line has one number, representing the vertex label.

Next m lines, each line has three numbers (a, b, 1), representing an edge between vertex a and vertex b.

You can also replace "\n" with ";" at the end of each line.

**Example 1**

1 10 0

10 

9

1

0

0

1

2

0

0

0

0

0

0 8 1

1 7 1

1 9 1

1 6 1

2 5 1

2 9 1

3 9 1

4 8 1

4 7 1

**Example 2**

1 10 0; 3; 2; 1; 0; 0; 0 1 1; 0 2 1;
