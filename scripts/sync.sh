cd "$(dirname "$0")"
WORKERS_FILE=../configs/workers

function sync_hosts {
    HVD_DIR=$HOME/.local/lib/python3.5/site-packages/horovod-0.19.1-py3.5-linux-x86_64.egg/
    APP_DIR=$HOME/workspace/tf-resnet-cifar10
    MPI_DIR=$HOME/local/openmpi-4.0.4/lib/

    while read line
    do
        HOST=$line
        echo sync with $HOST
        if [ -n "$HOST" ]; then
            echo "[INFO]    Sync '$HOST'"
            rsync -chazP --stats --delete $HVD_DIR/ $HOST:$HVD_DIR
            rsync -chazP --stats --delete $APP_DIR/*.py $HOST:$APP_DIR
            rsync -chazP --stats --delete $APP_DIR/files/ $HOST:$APP_DIR/files/
            rsync -chazP --stats --delete $APP_DIR/scripts/ $HOST:$APP_DIR/scripts/
            rsync -chazP --stats --delete $APP_DIR/configs/ $HOST:$APP_DIR/configs/
            rsync -chazP --stats --delete $MPI_DIR/ $HOST:$MPI_DIR
        fi
    done < ${WORKERS_FILE}
}

sync_hosts

