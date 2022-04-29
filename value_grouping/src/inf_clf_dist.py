"""Distributed classifier inference"""

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
import ray
import torch
from ray import tune
from scipy.special import softmax
from sentence_transformers.models import Transformer
from torch.utils.data import DataLoader

import utils
from contextualized_sbert.data import EntityContext, preprocess_clf_data
from contextualized_sbert.models import Classifier, EntityPooling, EntitySBERT

logger = logging.getLogger(__name__)

# must setup ray before hydra changes working directory
ray.init()

def run_clf_model_inference(model, tokenized_examples, batch_size=128):
  inf_loader = DataLoader(tokenized_examples, shuffle=False, batch_size=batch_size)
  inf_loader.collate_fn = model.smart_batching_collate
  model.eval()
  model.to(model._target_device)

  all_logits = None
  with torch.no_grad():
    last_progress = 0
    for i, batch in enumerate(inf_loader):
      # batch is a list of sentence features,
      # in this case we only have one
      sentence_features, _ = batch  # no labels
      output = model(sentence_features[0])
      logits = output["classifier_logits"]
      all_logits = logits.cpu() if all_logits is None else torch.cat([all_logits, logits.cpu()])

      progress = i / len(inf_loader)
      if progress - last_progress > 0.1:
        last_progress = progress
        tune.report(progress=f"{progress*100:.2f}%", stage="inf")
  return all_logits.numpy()


def run_clf_inference_on_pts(config):
  cfg = config["cfg"]  # hydra config
  pts = config["pts"]
  random.seed(cfg.rnd_seed)

  label_map = utils.Pkl.load(Path(cfg.model_dir, "label_map.pkl"))
  rev_label_map = {v: k for k, v in label_map.items()}
  if not cfg.try_load:
    # load models
    transformer = Transformer(cfg.model_dir)
    pooling = EntityPooling.load(Path(cfg.model_dir, "1_EntityPooling").as_posix())
    classifier = Classifier.load(Path(cfg.model_dir, "Clf_layer"))
    model = EntitySBERT(modules=[transformer, pooling, classifier])

    # read candidate generation results and group by pt
    tune.report(stage="prep")

    pt2range = dict()
    all_examples = []
    for pt in pts:
      candidate_docs = utils.JsonL.load(Path(cfg.candidate_dir, f"{pt}.chunk.jsonl"))

      # convert each phrase into an example
      inf_examples = []
      phrase2example_id = defaultdict(list)
      for doc in candidate_docs:
        for i, phrase in enumerate(doc):
          left_context = " ".join(doc[:i])
          if i + 1 < len(doc):
            right_context = " ".join(doc[i + 1:])
          else:
            right_context = ""
          entity_context = EntityContext(left_context, phrase, right_context)
          inf_examples.append((entity_context, pt))
          phrase2example_id[phrase].append(len(inf_examples) - 1)

      # print(pt, "#phrases", len(phrase2example_id))

      # sampling
      if cfg.sampling:
        sampled_examples = []
        for phrase, example_ids in phrase2example_id.items():
          if len(example_ids) > cfg.sampling:
            sampled_ids = random.sample(example_ids, cfg.sampling)
          else:
            sampled_ids = example_ids
          for sid in sampled_ids:
            sampled_examples.append(inf_examples[sid])
        inf_examples = sampled_examples

      # keep track of the range of pt
      start = len(all_examples)
      all_examples.extend(inf_examples)
      end = len(all_examples)

      pt2range[pt] = (start, end)
      # print(pt, start, end)

    # convert to tensors
    tokenized_samples, _ = preprocess_clf_data(all_examples, model.tokenizer, label_map=label_map, max_seq_length=model.max_seq_length, disable_tqdm=True)

    # run inference on examples
    all_logits = run_clf_model_inference(model, tokenized_samples, batch_size=cfg.batch_size)
    # all_probs = softmax(all_logits, axis=-1)

  # per pt post-processing
  tune.report(stage="post-process")
  for pt in pts:
    mask = []
    for label, idx in label_map.items():
      if not label.startswith(pt):
        mask.append(idx)
    mask = np.array(mask)

    n_class_pt = len(label_map) - len(mask)
    logger.debug(f"#class for pt={pt} is {n_class_pt}")
    if n_class_pt == 0:
      logger.warning(f"PT {pt} has zero valid class, skip inference")
      continue
    if not cfg.try_load:
      # print(pt, "start:end", start, end)
      start, end = pt2range[pt]

      logits_pt = all_logits[start:end, :]
      # logits_pt[:, mask] = 0.
      probs_pt = softmax(logits_pt, axis=-1)
      probs_pt[:, mask] = 0.
      examples_pt = all_examples[start:end]
      # print(probs_pt.shape)
      # print(probs_pt[0])
      # print(examples_pt[:2])

      utils.IO.ensure_dir(cfg.output_dir)
      # np.save(Path(cfg.output_dir, f"{pt}.probs.npy"), probs_pt)
      utils.Pkl.dump(examples_pt, Path(cfg.output_dir, f"{pt}.examples.pkl"))
    else:
      probs_pt = np.load(Path(cfg.output_dir, f"{pt}.probs.npy"))
      examples_pt = utils.Pkl.load(Path(cfg.output_dir, f"{pt}.examples.pkl"))

    # gather all the contextualized predictions
    def _parse_probs(probs, rev_label_map, thres):
      pred_class = np.argmax(probs)
      pred_prob = probs[pred_class]
      if pred_prob < thres:
        return -1
      return rev_label_map[pred_class]

    phrase2pred_labels = defaultdict(list)
    for example, probs in zip(examples_pt, probs_pt):
      pred_class = _parse_probs(probs, rev_label_map, 1 / n_class_pt / 2)
      if pred_class == -1:
        continue
      phrase = example[0].entity
      phrase2pred_labels[phrase].append(pred_class)
    # print(pt, "phrase2pred_labels", len(phrase2pred_labels))

    phrase2cluster = dict()
    for phrase, all_preds in phrase2pred_labels.items():
      # if phrase == "11 oz":
      #   print(phrase, all_preds)
      pred_counter = Counter(all_preds)
      most_freq_pred, sup = pred_counter.most_common(1)[0]
      if sup > 0.5 * len(all_preds):
        phrase2cluster[phrase] = most_freq_pred

    with Path(cfg.output_dir, f"{pt}.pred.txt").open("w") as f:
      for phrase, cluster in phrase2cluster.items():
        f.write(f"{phrase}\t{cluster}\n")


@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.inference.clf

  if cfg.selected_pt:
    pts = [s.strip() for s in cfg.selected_pt.split(",")]
  else:
    pts = []
    for candidate_file in Path(cfg.candidate_dir).glob("*.chunk.jsonl"):
      pt_name = candidate_file.stem[:-len(".chunk")]
      pts.append(pt_name)

  logger.info(f"Loaded {len(pts)} PTs")
  logger.info(f"First 3 PTs: {pts[:3]}")
  logger.info(f"n_gpu = {cfg.n_gpu}")
  logger.info(f"Group PTs into {cfg.n_gpu} groups")

  grouped_pts = [a.tolist() for a in np.array_split(pts, cfg.n_gpu)]
  tune.run(run_clf_inference_on_pts, config={
    "cfg": cfg,
    "dist": True,
    "pts": tune.grid_search(grouped_pts)
  }, resources_per_trial={"cpu": cfg.cpu_per_job, "gpu": 1})


if __name__ == "__main__":
  main()
