

for i in 1 2 3 4 5 6 7 8 9 
do
	echo $i
	python checkAcc.py ../data/Syn-BA/test/GT11.txt output/Syn-BA_t\="$i"_output.txt $i
	python checkAcc.py ../data/Syn-BA/test/GT11.txt output/Syn-BA_t\="$i"_candidate.txt $i

done
