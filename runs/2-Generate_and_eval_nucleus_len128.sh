
TASK=DITTO_openended
SAVE_LOG_PATH=/mnt/nfs-storage/jim/DITTO_chunk/checkpoints/evaluation_logs_nucleus

mkdir -p $SAVE_LOG_PATH

DS_SPLIT=test
bms=1
bnb=0
tpk=1
tpp=0.9
sttpk=1
sttpp=0


SAVE_PATH=/mnt/nfs-storage/jim/DITTO_chunk/checkpoints/$TASK
ls -l $SAVE_PATH
python -u /mnt/nfs-storage/jim/DITTO_chunk/fairseq/custom/evaluation_ar_chunked_data.py \
    --batch-size-single-prediction 1536 --batch-size-completion 48 \
    --data-prefix-length 50 --completion-length 100 \
    --save-path $SAVE_LOG_PATH --ckpt best \
    --model-path $SAVE_PATH \
    --data-split $DS_SPLIT \
    --beam-size $bms --beam-ngram-block $bnb --topp $tpp --topk $tpk --singletoken-topk $sttpk --singletoken-topp $sttpp \
    --data-dir /mnt/nfs-storage/jim/dataset/data-bin/chunked_wikitext-103/13-result-Rel_ver_9.0_preperocess_2/ \
    --base-dir ./


python -i report_metrics.py \
    --eval-dir $SAVE_LOG_PATH \
    --report-mauve \
    --model-names $TASK
