# Attribute Value Candidate Generation

This repo contains code for product title segmentation. Given an product title, e.g., "mccafe french roast k cup pods", we aim to segment it into phrases that can be attribute values, e.g., ["mccafe", "french roast", "k cup", "pods"].

There are two steps, **preprocessing** and **chunking**. The preprocessing step runs the product titles through a pre-trained BERT model to obtain word to word impact scores. The chunking step segments the titles into phrases based on the scores, and applies post processing to enhance phrase completeness.

## Running the code
### Dependencies
```
numpy
ray[tune]
python>=3.8
pytorch
transformers==4.6.1
tqdm
```

### Data
Input files should be jsonl files. For each product type (PT), the raw input should be stored in `[PT].jsonl`. Each line should have "asin" (unique ID) and "title" fields.

A pre-trained BERT checkpoint shall be loaded. Please use our in-domain fine-tuned language model [here](https://www.dropbox.com/s/8iijbdyinkxls34/oamine_bert.zip?dl=0).

In our dataset release, the files under `raw` directory is used as input for this part of the code. The output will look like files under `candidate` directory. Please use the BERT checkpoint that we release with the code.

### Running the code

The running script `preprocess_dist.sh` and `chunk_dist.sh` are for preprocessing and chunking respectively.
The scripts are for distributed running, where we create one job for each PT input and run it on one GPU.
You may use `preprocess.py` and `chunk.py` for single GPU jobs.

There will be two files generated for each PT, `[PT_name].chunk.jsonl` and `[PT_name].asin.txt`. The first file contains segmented product titles, and the second file contains aligned IDs.
