#!/bin/bash
dir=$1
c=32768
while [ $c -le 134217728 ] 
do 	
	./generate.exe $c > $dir/$c.in
	sleep 1.0
	((c=$c*2))
done
