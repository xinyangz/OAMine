# Attribute Value Grouping

**Please go to [candidate_generation](../candidate_generation/) sub directory and run the candidate generation step before following this guide.**
## Introduction
This repo contains code for attribute value grouping. Given attribute value candidates from segmented product titles, we leverage weak supervision, given by user as seed sets, to generate attribute value clusters. Each of the generated cluster represents a type of attributes.

There are three main steps for the grouping model:
1. Data generation
2. Attribute-aware multitask fine-tuning (binary + contrastive + classification)
3. Inference (w/ optional evaluation)

The first step takes segmented product titles, user given seed sets, to generate fine-tuning data for three objectives: binary meta-classfication, contrastive learning, and classification.

The second steps fine-tunes BERT representation using the following objectives:
1. Binary meta-classification. Training data is (value_1 + context_1, value_2 + context_2, label). Label shows if two values are from the same attribute. Positive labels generated from seed sets. Negative labels generated from both seed sets and product-level regularization.
2. Contrastive learning. Training data is (anchor + a_context, positive + p_context, negative + n_context). We apply a triplet loss that enforce embedding from anchor to positive value is closer than that from anchor to negative value. Triplets are derived from our binary data generation process.
3. Classification. Training data is (value + context + PT_name, PT_attribute). E.g., ("medium roast w/ context [SEP] coffee", "coffee_roast_type"). This objective directly puts each value into an attribute cluster.

The third step is inference. First, get embedding for each value candidate. Then, we run DBSCAN on the embedding to generate attribute clusters. After that, we run classifier inference. Finally, we combine results from DBSCAN and classifier.

## Running the code

### Dependencies
```
transformers==4.6.1
sentence_transformers==2.0.0
pytorch
hydra-core  % and its color log plugin, refer to https://hydra.cc/docs/intro/
flashtext  % for keyphrase matching
dataclasses_json
plac
ray[tune]
scikit-learn
tqdm
```

### Prepare data
Check our [dataset release](https://www.dropbox.com/s/1eksfr3k9iqbo32/amazon.zip?dl=0) and make sure the following subdirectories exist:
1. `candidate` contains output `*.chunk.jsonl` and `*.asin.txt` files from the previous candidate generation step.
2. `seed` contains seed sets. `[PT].seed.jsonl` has seed sets for one PT. Each line is an array, first element is the attribute name, second element is an array of attribute values.
3. `dev` contains dev set evaluation clusters. Follows the same format as seeds.
4. `test` contains test set clusters. Follows the same format as seeds.

An in-domain fine-tuned BERT model is preferred. Please use [our checkpoint](https://www.dropbox.com/s/8iijbdyinkxls34/oamine_bert.zip?dl=0).

Once the input files are placed, go to `exp_config/dataset/amazon.yaml` and set the experiment folder. Go to `exp_config/tuning/multitask.yaml` and set pre-trained BERT directory.

### Running experiments
Go to the top level folder. Check `run_iterative.sh` for terative training. Update paths in the `.sh` files appropriately, and then the experiments are ready to go.

### Output
The output will be generated to the `exp_dir` folder set in the `.sh` files or in the config files. Each iteration has its own folder. `data` contains fine-tuning data for that iteration. `model` contains fine-tuned model. `emb_inf` contains embedding after fine-tuning. `clf_inf` contains classifier inference results. `ensemble_inf` contains the predictions.

### Code organization
Please follow `run_iterative.sh` to see what files are invoked for each step of the framework.

`src` contains Python code, `exp_config` contains hyperparameters and experiment configurations.

The config files use [hydra](https://hydra.cc/). The main config file is `exp_config/config.yaml`. It contains default options for each step, and configurations for a single experiment (`run:...`).

The `dataset` subfolder contains path to dataset. Multiple datasets can be given. Only the one specified in `config.yaml` at top level will be used.

The `preprocessing`, `tuning`, `inference` subfolders contain configurations for each step of the framework.

Please refer to the [hydra package](https://hydra.cc/) for how they work together.

