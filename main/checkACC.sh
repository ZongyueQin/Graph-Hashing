

for i in 1 2 3 4 5 6 #7 8 9 
do
	echo $i
	python3 checkAcc.py ~/ans/FULL_ALCHEMY_ans.txt output/0624_FULL_ALCHEMY_t\="$i"_output.txt $i
	python3 checkAcc.py ~/ans/FULL_ALCHEMY_ans.txt output/0624_FULL_ALCHEMY_t\="$i"_candidate.txt $i

done
