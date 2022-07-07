#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

export CUDA_VISIBLE_DEVICES=1
NUM_GPU=1
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
python3 train.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file "./data/pretrain.train.csv" \
    --validation_file "./data/pretrain.test.csv" \
    --output_dir ./models/result_recall_pretrain \
    --num_train_epochs 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --metric_for_best_model eval_acc \
    --learning_rate 3e-5 \
    --max_seq_length 113 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_mlm \
    "$@"
