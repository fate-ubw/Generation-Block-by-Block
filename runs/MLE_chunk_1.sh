
SAVE_DIR=/mnt/nfs-storage/jim/DITTO_chunk/checkpoints/baseline_model
mkdir -p $SAVE_DIR
export HOME=/mnt/nfs-storage/jim/DITTO_chunk
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export device='cuda'
#unset WORLD_SIZE
unset MASTER_PORT
unset RANK
unset MASTER_ADDR
python -u /mnt/nfs-storage/jim/DITTO_chunk/train.py \
 --task language_modeling_with_generation /mnt/nfs-storage/jim/DITTO_chunk/datas/data-bin/chunked_wikitext-103/13-result-Rel_ver_9.0_preperocess_2 \
 --user-dir ./fairseq/custom --arch transformer_lm_ul_base --max-tokens 128 --tokens-per-sample 128 \
 --fp16  --fp16-scale-tolerance=0.25 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \
 --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
 --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \
 --keep-interval-updates 2 --no-progress-bar --log-interval 100 \
 --criterion cross_entropy_wcustom_metrics \
 --save-dir $SAVE_DIR \
 --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt
