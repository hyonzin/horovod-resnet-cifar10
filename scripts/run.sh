#!/bin/bash
cd "$(dirname "$0")"/..
source scripts/configs.sh
############################
#for n in {15..1};do
#    echo $i
#    run $NONE $n
#    run $FP16 $n
#    run $CPU $n
#    run $GPU $n
#    run $AllreduceAdacomp $n
#done

n=4
#run $FP16 $n
#run $CPU $n
#run $GPU $n
run $AllreduceAdacomp $n
 
