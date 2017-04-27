dir1=$1
dir2=$2

#./scripts/exec.sh sorts/bitonic/bitonicsort.exe $dir1 > $dir2/bitonicsort.time
#./scripts/exec.sh sorts/mergesort/mergesort.exe $dir1 > $dir2/mergesort.time
#./scripts/exec.sh sorts/oddevensort/oddevensort.exe $dir1 > $dir2/oddevensort.time
#./scripts/exec.sh sorts/quicksort/quicksort $dir1 > $dir2/quicksort.time
./scripts/exec.sh radixcub.exe $dir1 > $dir2/radixcub.time
./scripts/exec.sh radixthrust.exe $dir1 > $dir2/radixthrust.time
#./scripts/exec.sh mergemgpu.exe $dir1 > $dir2/mergemgpu.time
#./scripts/exec.sh radixthrust_stable.exe $dir1 > $dir2/radixthrust_stable.time




