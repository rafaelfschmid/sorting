#!/bin/bash
c=8
while [ $c -le 524288 ]
do
	#echo $c 
	d=2
	((x=$d*$c))
	while [ $x -le 1048576 ] 
	do 	
		#echo $d
		      for b in `seq 1 10`; do
		           #mpirun --machinefile $PBS_NODEFILE -np 256 ./seqmax21_edson
		           ./diff.exe $c $d > inputs/$c"_"$d.in
		      done
		      #echo " "
		((d=$d*2))
		((x=$d*$c))
	done
	((c=$c*2))
done
