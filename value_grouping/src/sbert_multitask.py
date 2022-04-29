"""Multitask representation fine-tuning"""
import logging
import math
import random
from pathlib import Path

import hydra
import torch
from sentence_transformers import losses
from sentence_transformers.models import Transformer
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

import utils
from contextualized_sbert.data import (load_binary_dataset, load_clf_dataset,
                                       load_triplet_dataset)
from contextualized_sbert.models import (ClassifierClfLoss,
                                         ContextualizedBinaryClfEvaluator,
                                         EntityPooling, EntitySBERT)

#### Code to print debug information to stdout
utils.Log.config_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.tuning

  torch.cuda.set_device(cfg.gpu)

  use_triplet = cfg.use_triplet
  use_clf = cfg.use_clf

  random.seed(cfg.rnd_seed)
  utils.IO.ensure_dir(cfg.output_dir)

  # Convert the dataset to a DataLoader ready for training

  train_len = 0
  tokenizer = BertTokenizerFast.from_pretrained(cfg.pretrained_model)
  train_file = Path(cfg.data_dir, "train_binary.jsonl")
  binary_dataset, binary_dev_dataset = load_binary_dataset(train_file,
                                                           tokenizer,
                                                           train_limit=None,
                                                           dev_limit=None,
                                                           dev_file=Path(cfg.data_dir, "dev_binary.jsonl"),
                                                           max_seq_length=cfg.max_seq_length)
  binary_dataloader = DataLoader(binary_dataset, shuffle=True, batch_size=cfg.train_batch_size)
  train_len += len(binary_dataloader)
  logger.info(f"Binary training dataset size: {len(binary_dataset)}")
  logger.info(f"Dev set size: {len(binary_dev_dataset)}")

  if use_triplet:
    triplet_dataset = load_triplet_dataset(Path(cfg.data_dir, "train_triplet.jsonl"),
                                           tokenizer,
                                           train_limit=None,
                                           dev_file=None,
                                           max_seq_length=cfg.max_seq_length)
    triplet_dataloader = DataLoader(triplet_dataset, shuffle=True, batch_size=cfg.train_batch_size)
    logger.info(f"Triplet training dataset size: {len(triplet_dataset)}")
    train_len += len(triplet_dataloader)

  if use_clf:
    clf_dataset, label_map = load_clf_dataset(Path(cfg.data_dir, "train_clf.jsonl"),
                                              tokenizer,
                                              dev_file=None,
                                              max_seq_length=cfg.max_seq_length)
    clf_dataloader = DataLoader(clf_dataset, shuffle=True, batch_size=cfg.train_batch_size)
    logger.info(f"Classification training dataset size: {len(clf_dataset)}")
    train_len += len(clf_dataloader)
    utils.Pkl.dump(label_map, Path(cfg.output_dir, "label_map.pkl"))

  # Development set: Measure correlation between cosine score and gold labels
  logging.info("Read dev dataset")
  evaluator = ContextualizedBinaryClfEvaluator.from_input_examples(binary_dev_dataset, name="dev")

  # Load a pre-trained sentence transformer model
  transformer = Transformer(cfg.pretrained_model)
  pooling = EntityPooling(transformer.get_word_embedding_dimension())
  model = EntitySBERT(modules=[transformer, pooling])
  model.max_seq_length = cfg.max_seq_length

  train_objectives = []
  binary_train_loss = losses.CosineSimilarityLoss(model=model)
  train_objectives.append((binary_dataloader, binary_train_loss))
  if use_triplet:
    triplet_train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE)
    train_objectives.append((triplet_dataloader, triplet_train_loss))
  if use_clf:
    clf_train_loss = ClassifierClfLoss(model, transformer.get_word_embedding_dimension(), len(label_map), cfg.clf_dropout)
    train_objectives.append((clf_dataloader, clf_train_loss))
    model.add_save_component(clf_train_loss, "Clf_layer")


  # Configure the training. We skip evaluation in this example
  warmup_steps = math.ceil(train_len * cfg.num_epochs * 0.1)  #10% of train data for warm-up
  logging.info("Warmup-steps: {}".format(warmup_steps))

  # Train the model
  model.fit(train_objectives=train_objectives,
            evaluator=evaluator,
            epochs=cfg.num_epochs,
            evaluation_steps=cfg.evaluation_steps,
            warmup_steps=warmup_steps,
            use_amp=True,
            output_path=cfg.output_dir,
            checkpoint_save_total_limit=cfg.checkpoint_save_total_limit,
            checkpoint_save_steps=cfg.checkpoint_save_steps,
            checkpoint_path=Path(cfg.checkpoint_path).as_posix())


if __name__ == "__main__":
  main()
