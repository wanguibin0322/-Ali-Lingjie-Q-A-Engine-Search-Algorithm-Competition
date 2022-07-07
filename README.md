比赛地址：https://tianchi.aliyun.com/competition/entrance/531946/introduction
1. 召回模型预训练
	1.1 数据源
		a. 数据源1：data目录下：corpus.tsv train.query.txt qrels.train.tsv dev.query.txt 
			来源：比赛官网提供
		b. 数据源2：data/ecom/目录下: corpus.tsv       dev.query.txt    qrels.dev.tsv    qrels.train.tsv  train.query.txt
			来源：https://github.com/Alibaba-NLP/Multi-CPR
			data/ecom
		c. 数据源3：data/big_corpus/目录下: big_corpus.txt
			来源：https://github.com/FeiSun/ProductTitleSummarizationCorpus
			cd big_corpus
			cat big_corpus.tar.gz_* > big_corpus.tar.gz
			tar zxvf big_corpus.tar.gz
	1.2 数据加工过程：
		准备好三个数据源数据后运行以下脚本，可在data下生成： pretrain.test.csv   pretrain.train.csv
		#python3 gen_recall_pretrain_data.py		
	1.4 预训练脚本(比较费时，a300系列机器预计跑2-3天)：
		#source run_recall_pretrain.sh 
2. 召回模型训练
	2.0 参考代码： https://github.com/enze5088/WenTianSearch
	2.1 数据源:data目录下：corpus.tsv train.query.txt qrels.train.tsv dev.query.txt
	        来源：比赛官网提供
	2.2 数据加工过程：data下生成： query.doc.test.csv query.doc.csv
		#python3 gen_base_recall_data.py
	2.3 训练模型1脚本：
		#source run_recall_base.sh
	2.4 训练模型2脚本：
		#python3 get_recall_data.py
		#source run_recall.sh
3. 排序模型预训练(tf1.12)：
	3.1 数据源：同1.1
	3.2 数据加工过程：
		#python3 gen_rank_pretrain_data.py
	3.3 预训练脚本(先下载chinese_roberta_wwm_ext_tensorflow: https://drive.google.com/file/d/1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt/view?usp=drive_open)
		模型位置： models/chinese_roberta_wwm_ext_tensorflow
		#cd bert-master
		#source train.sh
4. 排序模型训练(tf1.12)：
	4.0 参考代码，官网baseline，见比赛地址
	4.1 数据源：data目录下： tokenize/
		#python3 gen_rank_data.py
	4.2 训练脚本：
		#python3 trainer.py
5. 数据打包 
	#python3 wrapper.py
	#cd data/feature_recall/
	#source tar.sh
6. 数据预测：
	#source predict.sh
		
