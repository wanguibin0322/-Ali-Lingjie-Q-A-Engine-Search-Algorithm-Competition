import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                desc=f"Tokenize: {os.path.basename(input_file)}"):
            try:
                seq_id, text = line.split("\t")
            except:
                import pdb
                pdb.set_trace()
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            outFile.write(json.dumps(
                {"qid":seq_id, "input_ids":ids}
            ))
            outFile.write("\n")
    

def tokenize_queries(args, tokenizer):
    for mode in ["dev","train"]:
        query_output = f"{args.output_dir}/{mode}.query.json"
        tokenize_file(tokenizer, f"{args.msmarco_dir}/{mode}.query.txt", query_output)


def tokenize_collection(args, tokenizer):
    collection_output = f"{args.output_dir}/corpus.json"
    tokenize_file(tokenizer, f"{args.msmarco_dir}/corpus.tsv", collection_output)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/tokenize")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenize_queries(args, tokenizer)  
    tokenize_collection(args, tokenizer) 
