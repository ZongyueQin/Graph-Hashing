

for i in 1 2 3 4 5
do
	python checkAcc.py ../data/AIDS/test/GT11.txt output/AIDS_k5_t\="$i"_output.txt $i
done
