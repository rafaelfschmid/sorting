#!/bin/bash
prog1=$1
prog2=$2
dir=$3

for filename in `ls -tr $dir`; do
		file=$filename
		c=$(echo $file| cut -d'.' -f 1)
		echo $c".in"

	./$prog1 < $dir/$filename > "test1.txt"
	./$prog2 < $dir/$filename > "test2.txt"

	if ! cmp -s "test1.txt" "test2.txt"; then
		echo "There are something wrong."
		break;
	else
		echo "Everthing ok."		
	fi
done
