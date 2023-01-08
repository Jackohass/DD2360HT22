#!/bin/bash
ARRAYSIZE=(4096 16384 65536 131070 262144 1048576)
ARRAYSEG=(32 256 1024 2048 4096)
SIZEELEMENTS=${#ARRAYSIZE[@]}
SEGELEMENTS=${#ARRAYSEG[@]}

for (( i=0;i<$SEGELEMENTS;i++)); do
    	echo Running streamed addvec with seg size: ${ARRAYSEG[${i}]} $'\n'
	for (( j=0;j<$SIZEELEMENTS;j++)); do
		echo With input length ${ARRAYSIZE[${j}]}
        	./code/lab3_ex1Stream.o ${ARRAYSIZE[${j}]} ${ARRAYSEG[${i}]}
		echo $'\n'
	done
	echo $'\n'
done 
