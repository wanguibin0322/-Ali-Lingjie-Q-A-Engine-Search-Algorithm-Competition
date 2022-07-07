BERT_BASE_DIR='../models/chinese_roberta_wwm_ext_tensorflow'
python3 create_pretraining_data.py \
  --input_file=../data/rank_pretrain_data/rank_pretrain.txt \
  --output_file=../models/result_rank_pretrain/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=113 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
