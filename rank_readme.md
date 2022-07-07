#  
- data.py: for processing data
- rank_model.py: model for training
- trainer.py: for training model 
- wrapper.py: modify the trained model's inputs and output to the format of the competition, and convert a checkpoint to saved model  
- bert: bert model

- simple_rank_model.py: an example of a simple rank model (not bert-like). Note that this is just model, no training. If you want to train it, you should implement trainer.py by yourself.
- wrapper_simple_model.py: wrap the simple rank model in simple_rank_model.py 


## training
### data
   - We provide example in directory **data**.
### training scripts
   - use `python trainer.py --arguments xx`

## convert a checkpoint model to saved_model and wrap the model to meet the requirements of the competition
   - use `python wrapper.py --arguments xx`

## Example
There is an example to show how to use simple_rank_model.py and wrapper_simple_model.py
Step 1. python simple_rank_model.py 
   - Its output is checkpoint in directory `temp`
Step 2. python wrapper_simple_model.py --ckpt_to_convert temp --output_dir temp.out1 --max_seq_length 100 
   - Its inputs are checkpoint_path, saved_model path and max_seq_length of your model
   - Its output is saved_model, which you can submit to the competition.
** Note **
If you want to design your own model, you need to modify `Class Ranker` in `simple_rank_model.py`. Then, you also need to modify `class RankModel` in `wrapper_simple_model.py`.