#!/bin/bash
dir=$1
c=8
while [ $c -le 33554432 ] 
do 	
	./generate.exe $c > $dir/$c.in
	sleep 0.5
	((c=$c*2))
done
