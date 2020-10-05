#!/bin/bash
cd "$(dirname "$0")"

source configs.sh
############################
#run $NONE 1
#run $CPU  3
#run $GPU  15

#for i in {15..1};do
#    echo $i
#    run $NONE $i
#    run $CPU $i
#    run $GPU $i
#done


#for i in {2..10..2};do
#    echo 
#    echo $i
#    run $NONE $i
#    run $CPU $i
#    run $GPU $i
#done

n=1
#run $NONE $n
#run $CPU $n
run $GPU $n
 
