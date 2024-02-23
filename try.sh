num=1
cat ./result/cse545/gen_list.txt  | while read line
do
	echo $num
	cat ./result/cse545/result.txt | grep $line
	num=$(($num+1))
done