#!/bin/bash
ARRAYSIZE=(100 200 500 1000 2000 5000 10000) # 20000)
ARRAYSEG=(128) # (32 128 256 1024 2048)
SIZEELEMENTS=${#ARRAYSIZE[@]}
SEGELEMENTS=${#ARRAYSEG[@]}
for((i=0;i<$SEGELEMENTS;i++));do 
echo Running heat calc with dimx: ${ARRAYSEG[${i}]} $'\n'
for((j=0;j<$SIZEELEMENTS;j++));
do echo With nstep: ${ARRAYSIZE[${j}]}
./code/lab4-ex4.o ${ARRAYSEG[${i}]} ${ARRAYSIZE[${j}]}
echo $'\n'
done
echo $'\n'
done