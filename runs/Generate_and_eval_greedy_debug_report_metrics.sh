
TASK=DITTO_openended
SAVE_LOG_PATH=/mnt/nfs-storage/jim/DITTO_chunk/checkpoints/evaluation_logs

mkdir -p $SAVE_LOG_PATH

DS_SPLIT=test
bms=1
bnb=0
tpk=1
tpp=0
sttpk=1
sttpp=0
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 

SAVE_PATH=/mnt/nfs-storage/jim/DITTO_chunk/checkpoints/$TASK
ls -l $SAVE_PATH

python  report_metrics_no_chunk_stamp.py \
    --eval-dir $SAVE_LOG_PATH \
    --report-mauve \
    --model-names $TASK
