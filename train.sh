#预训练
#准备好README中的三个数据源后
python3 gen_recall_pretrain_data.py		
#source run_recall_pretrain.sh 

#预训练结束后 模型位置： models/result_recall_pretrain
python3 gen_base_recall_data.py
#source run_recall_base.sh
#训练模型2脚本：
python3 get_recall_data.py
#source run_recall.sh

#排序模型预训练(tf1.12)：
python3 gen_rank_pretrain_data.py
#cd bert-master
#source train.sh

python3 gen_rank_data.py
#python3 trainer.py
python3 wrapper.py

#上传打包
#cd data/feature_recall/
#source tar.sh

		
