#!/bin/bash
cd "$(dirname "$0")"/..

NONE="none"
FP16="fp16"
CPU="adacompCPU"
GPU="adacompGPU"
AllreduceAdacomp="allreduceAdacomp"

NODES_1="node01:1 "
NODES_2="node01:1,node02:1 "
NODES_3="node01:1,node02:1,node03:1 "
NODES_4="node01:1,node02:1,node03:1,node04:1 "
NODES_5="node01:1,node02:1,node03:1,node04:1,node05:1 "
NODES_6="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1 "
NODES_7="node01:1,node02:1,node03:1,node04:1,node06:1,node07:1,node08:1 "
NODES_8="node01:1,node02:1,node03:1,node04:1,node06:1,node07:1,node08:1,node09:1 "

NODES_9="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1 "
NODES_10="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1 "
NODES_11="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1 "
NODES_12="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1,node12:1 "
NODES_13="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1,node12:1,node13:1 "
NODES_14="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1,node12:1,node15:1,node14:1 "
NODES_15="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1,node12:1,node13:1,node14:1,node15:1 "
NODES_16="node01:1,node02:1,node03:1,node04:1,node05:1,node06:1,node07:1,node08:1,node09:1,node10:1,node11:1,node12:1,node13:1,node14:1,node16:1 "

NODES=("" ${NODES_1} ${NODES_2} ${NODES_3} ${NODES_4} ${NODES_5} \
    ${NODES_6} ${NODES_7} ${NODES_8} ${NODES_9} ${NODES_10} \
    ${NODES_11} ${NODES_12} ${NODES_13} ${NODES_14} ${NODES_15} )

run()
{
COM_OP=${1:-fp16}
NUM_NODE=${2:-1}
K=${3:-1}
LOG="./log/log_${COM_OP}.txt"

date | tee -a ${LOG}

BIN="${HOME}/local/openmpi-4.0.4/bin/mpirun"

OP="${OP} -np ${NUM_NODE} -H ${NODES[$NUM_NODE]} "
#OP="${OP} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH "
#OP="${OP} -mca btl_openib_allow_ib true"
#OP="${OP} -mca btl_openib_allow_ib false -mca btl_tcp_if_include eth0"

#ethernet
#OP="${OP} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca btl_openib_allow_ib false --mca btl_tcp_if_include eth0"
#ib?
#OP="${OP} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH --mca btl_openib_allow_ib true --mca btl_tcp_if_include eth0"

OP="${OP} -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH"

# Ethernet
#OP="$OP --mca btl_openib_allow_ib false  --mca btl_tcp_if_include eth0"

# Infiniband
OP="$OP --mca btl_openib_allow_ib true --mca btl_openib_if_include mlx4_0:1 --mca pml ^ucx"



APP=run_trainer.py

APP_OP="${APP_OP} --compression=${COM_OP} \
    --k=${K} \
    --num_layers=56 \
    --data_path=files/cifar-10-batches-bin"

CMD="${BIN} ${OP} python3 ${APP} ${APP_OP}"

echo ${CMD} | tee -a ${LOG}
${CMD} | tee -a ${LOG}

#cat ~/.hzprof | awk '{system("ssh -p "$2" "$1" '$HOME'/workspace/c/hzprof/merge/merge.sh '${COM_OP}'_'${NUM_NODE}'nodes")}'
}

