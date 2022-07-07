BERT_BASE_DIR='../models/chinese_roberta_wwm_ext_tensorflow'
python3 run_pretraining.py \
  --input_file=../models/result_rank_pretrain/tf_examples.tfrecord \
  --output_dir=../models/result_rank_pretrain \
  --do_train=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=64 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=650000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
