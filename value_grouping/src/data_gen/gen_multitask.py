"""Combine binary, triplet, and classification data. Balance their size by sampling, and
generate data for single task ablations
See config/preprocessing/binary for configurations."""
import sys
from pathlib import Path

sys.path.insert(1, Path(sys.path[0]).parent.as_posix())  # to include parent dir imports

import logging
import random
from collections import Counter

import hydra
import numpy as np
import utils
from sklearn.model_selection import train_test_split

logger = logging.getLogger()


def down_sample(examples, n_sample, log_name="", stratify_key=None):
  # sample triplet, using random sampling
  if stratify_key is None:
    if len(examples) > n_sample:
      logger.info(f"Down sampling {log_name}")
      random.shuffle(examples)
      ret = examples[:n_sample]
    else:
      logger.warning(f"{log_name} set < requested size, keeping all")
      ret = examples
    return ret
  else:
    # sample binary ,using stratified sampling
    if len(examples) > n_sample:
      logger.info(f"Down sampling {log_name}")
      labels = [example[stratify_key] for example in examples]
      # keep examples with only 1 sample
      counter = Counter(labels)
      keep_examples = []
      split_labels = []
      split_examples = []
      for example in examples:
        label = example[stratify_key]
        if counter[label] < 2:
          keep_examples.append(example)
        else:
          split_examples.append(example)
          split_labels.append(label)
      ret, _= train_test_split(split_examples, train_size=n_sample, stratify=split_labels)
      ret.extend(keep_examples)
    else:
      logger.warning(f"{log_name} set < requested size, keeping all")
      ret = examples
    return ret


def triplet_to_binary(triplet_data):
  ret = []
  for example in triplet_data:
    anchor = example["anchor"]
    pos = example["positive"]
    neg = example["negative"]
    ret.append({
      "entity_a": anchor,
      "entity_b": pos,
      "label": 1
    })
    ret.append({
      "entity_a": anchor,
      "entity_b": neg,
      "label": 0
    })
  return ret


@hydra.main(config_path="../../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.preprocessing.multitask
  random.seed(cfg.rnd_seed)
  np.random.seed(cfg.rnd_seed)

  logger.info("Loading data")

  binary_dir = Path(cfg.binary_dir)
  output_dir = Path(cfg.output_dir)
  logger.info("binary")
  binary_train = utils.JsonL.load(Path(binary_dir, "train.jsonl"))
  binary_dev = utils.JsonL.load(Path(binary_dir, "dev.jsonl"))

  logger.info("triplet")
  triplet_dir = Path(cfg.triplet_dir)
  triplet_train = utils.JsonL.load(Path(triplet_dir, "train.jsonl"))
  triplet_dev = utils.JsonL.load(Path(triplet_dir, "dev.jsonl"))

  if not cfg.ablation:
    clf_dir = Path(cfg.clf_dir)
    logger.info("clf")
    clf_train = utils.JsonL.load(Path(clf_dir, "train.jsonl"))
    clf_dev = utils.JsonL.load(Path(clf_dir, "dev.jsonl"))
  else:
    clf_train = []
    clf_dev = []



  logger.info(f"Loaded training data: {len(triplet_train)} triplet, {len(binary_train)} binary, {len(clf_train)} clf")
  logger.info(f"Loaded dev data: {len(triplet_dev)} triplet, {len(binary_dev)} binary, {len(clf_dev)} clf")

  ### Down sampling and save multitask data
  logger.warning(f"Down sampling to {cfg.n_train} training examples per task")
  # down sample
  binary_train = down_sample(binary_train, cfg.n_train, log_name="binary", stratify_key="label")
  triplet_train = down_sample(triplet_train, cfg.n_train, log_name="triplet", stratify_key=None)
  if not cfg.ablation:
    clf_train = down_sample(clf_train, cfg.n_train, log_name="clf", stratify_key="label")

  logger.warning(f"Down sampling to {cfg.n_dev} dev examples per task")
  # down sample
  binary_dev = down_sample(binary_dev, cfg.n_dev, log_name="binary", stratify_key="label")
  triplet_dev = down_sample(triplet_dev, cfg.n_dev, log_name="triplet", stratify_key=None)
  if not cfg.ablation:
    clf_dev = down_sample(clf_dev, cfg.n_dev, log_name="clf", stratify_key="label")

  utils.IO.ensure_dir(output_dir)
  if not cfg.ablation:
    utils.JsonL.save(Path(output_dir, "train_binary.jsonl"), binary_train)
    utils.JsonL.save(Path(output_dir, "train_triplet.jsonl"), triplet_train)
    utils.JsonL.save(Path(output_dir, "train_clf.jsonl"), clf_train)
    utils.JsonL.save(Path(output_dir, "dev_triplet.jsonl"), triplet_dev)
    utils.JsonL.save(Path(output_dir, "dev_clf.jsonl"), clf_dev)
  else:
    ### generate matching data for ablation
    binary_from_triplet_train = triplet_to_binary(triplet_train)
    utils.JsonL.save(Path(output_dir, "train_binary.jsonl"), binary_train + binary_from_triplet_train)
  utils.JsonL.save(Path(output_dir, "dev_binary.jsonl"), binary_dev)



if __name__ == "__main__":
  main()
