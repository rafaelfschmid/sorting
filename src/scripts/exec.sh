#!/bin/bash
prog=$1
dir=$2
for filename in `ls -tr $dir`; do
	file=$filename
	file=$(echo $file| cut -d'.' -f 1)
	echo $file
  for b in `seq 1 10`; do
		./$prog < $dir/$filename
	done
	echo " "
done

