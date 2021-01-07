#!/usr/bin/env bash
#GPU_ID=$0,1,2,3,4,5,6,7
#CUDA_VISIBLE_DEVICES=1 python ./tools/train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py \
    --id 'sgmn' \
    --learning_rate 1e-4 \
    --word_drop_out 0.2 \
    --rnn_drop_out 0.2 \
    --jemb_drop_out 0.2 \
    --elimination True \
    --batch_size 64 \
    --model_method 'sgmn' \
    --workers 2