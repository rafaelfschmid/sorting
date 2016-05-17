./scripts/exec.sh sorts/bitonic/bitonicsort.exe ~/inputs/sorts > times/bitonicsort.time
./scripts/exec.sh sorts/mergesort/mergesort.exe ~/inputs/sorts > times/mergesort.time
./scripts/exec.sh sorts/oddevensort/oddevensort.exe ~/inputs/sorts > times/oddevensort.time
./scripts/exec.sh sorts/quicksort/quicksort ~/inputs/sorts > times/quicksort.time
./scripts/exec.sh radixcub.exe ~/inputs/sorts > times/radixcub.time
./scripts/exec.sh radixthrust.exe ~/inputs/sorts > times/radixthrust.time


