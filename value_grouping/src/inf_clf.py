"""Classifier inference"""

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
import pickle as pkl

import hydra
import numpy as np
import torch
from ray import tune
from scipy.special import softmax
from sentence_transformers.models import Transformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils
from contextualized_sbert.data import EntityContext, preprocess_clf_data
from contextualized_sbert.models import Classifier, EntityPooling, EntitySBERT

logger = logging.getLogger(__name__)

def run_clf_model_inference(model, tokenized_examples, batch_size=128, dist=False):
  inf_loader = DataLoader(tokenized_examples, shuffle=False, batch_size=batch_size)
  inf_loader.collate_fn = model.smart_batching_collate
  model.eval()
  model.to(model._target_device)

  all_logits = None
  with torch.no_grad():
    last_progress = 0
    for i, batch in enumerate(tqdm(inf_loader, desc="Inference", disable=dist)):
      # batch is a list of sentence features,
      # in this case we only have one
      sentence_features, _ = batch  # no labels
      output = model(sentence_features[0])
      logits = output["classifier_logits"]
      all_logits = logits.cpu() if all_logits is None else torch.cat([all_logits, logits.cpu()])

      progress = i / len(inf_loader)
      if dist and progress - last_progress > 0.1:
        last_progress = progress
        tune.report(progress=f"{progress*100:.2f}%", stage="inf")
  return all_logits.numpy()


def run_clf_inference_on_pts(config):
  cfg = config["cfg"]  # hydra config
  pts = config["pts"]
  dist = config["dist"]
  random.seed(cfg.rnd_seed)

  label_map = utils.Pkl.load(Path(cfg.model_dir, "label_map.pkl"))
  rev_label_map = {v: k for k, v in label_map.items()}

  logger.info("Loading model")
  transformer = Transformer(cfg.model_dir)
  pooling = EntityPooling.load(Path(cfg.model_dir, "1_EntityPooling").as_posix())
  classifier = Classifier.load(Path(cfg.model_dir, "Clf_layer"))
  model = EntitySBERT(modules=[transformer, pooling, classifier])

  # read candidate generation results and group by pt
  if dist:
    tune.report(stage="prep")

  pt2range = dict()
  all_examples = []
  for pt in tqdm(pts, desc="Prep. Examples", disable=dist):
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
    logger.info(f"{pt} {end - start} examples")

    pt2range[pt] = (start, end)
  logger.info(f"{len(all_examples)} in total")

  # convert to tensors
  logger.info("Preprocessing")
  tokenized_samples, _ = preprocess_clf_data(all_examples, model.tokenizer, label_map=label_map, max_seq_length=model.max_seq_length, disable_tqdm=dist)

  # run inference on examples
  logger.info("Run model inference")
  logger.info(f"{len(tokenized_samples)} examples")
  all_logits = run_clf_model_inference(model, tokenized_samples, batch_size=cfg.batch_size)

  # per pt post-processing
  logger.info("Post processing: gathering")
  if dist:
    tune.report(stage="post-process")
  for pt, (start, end) in tqdm(pt2range.items(), desc="Post Proc.", disable=dist):
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
    start, end = pt2range[pt]

    logits_pt = all_logits[start:end, :]
    # logits_pt[:, mask] = 0.
    probs_pt = softmax(logits_pt, axis=-1)
    if mask:
      probs_pt[:, mask] = 0.
    examples_pt = all_examples[start:end]


    all_probs = softmax(all_logits, axis=-1)
    if mask:
      all_probs[:, mask] = 0.

    utils.IO.ensure_dir(cfg.output_dir)
    np.save(Path(cfg.output_dir, f"{pt}.probs.npy"), all_probs)
    with open(Path(cfg.output_dir, f"{pt}.examples.pkl"), "wb") as f:
      pkl.dump(examples_pt, f)

    # gather all the contextualized predictions
    def _parse_probs(probs, rev_label_map, thres):
      pred_class = np.argmax(probs)
      pred_prob = probs[pred_class]
      if pred_prob < thres:
        return -1
      return rev_label_map[pred_class]

    phrase2pred_labels = defaultdict(list)
    phrase_prob = []
    for example, probs in zip(examples_pt, probs_pt):
      pred_class = _parse_probs(probs, rev_label_map, 1 / n_class_pt / 10)
      if pred_class == -1:
        continue
      phrase = example[0].entity
      prob = np.max(probs)
      phrase2pred_labels[phrase].append(pred_class)
      phrase_prob.append((phrase, pred_class, prob))

    phrase2cluster = dict()
    for phrase, all_preds in phrase2pred_labels.items():
      pred_counter = Counter(all_preds)
      most_freq_pred, sup = pred_counter.most_common(1)[0]
      if sup > 0.7 * len(all_preds):
        phrase2cluster[phrase] = most_freq_pred

    with Path(cfg.output_dir, f"{pt}.pred.txt").open("w") as f:
      for phrase, cluster in phrase2cluster.items():
        f.write(f"{phrase}\t{cluster}\n")

    cluster2phrases = defaultdict(list)
    for phrase, cluster, prob in phrase_prob:
      cluster2phrases[cluster].append((phrase, prob))
    # sort and dedup
    cluster_docs = []
    for cluster, phrase_probs in cluster2phrases.items():
      phrase_probs = list(sorted(phrase_probs, key=lambda x: x[-1], reverse=True))
      # dedup
      phrase_list = []
      phrase_set = set()
      for phrase, prob in phrase_probs:
        if phrase not in phrase_set:
          phrase_set.add(phrase)
          phrase_list.append(phrase)
      cluster_docs.append([cluster, phrase_list])

    # for phrase, cluster in phrase2cluster.items():
    #   cluster2phrases[cluster].append(phrase)
    # cluster2phrases = list(cluster2phrases.items())
    utils.JsonL.save(Path(cfg.output_dir, f"{pt}.fmt.pred.txt"), cluster_docs)


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
  # tune.run(run_clf_inference_on_pts, config={
  #   "cfg": cfg,
  #   "pts": tune.grid_search(grouped_pts)
  # }, resources_per_trial={"cpu": cfg.cpu_per_job, "gpu": 1})

  for pts in grouped_pts:
    config = dict()
    config["cfg"] = cfg
    config["dist"] = False
    config["pts"] = pts
    run_clf_inference_on_pts(config)

if __name__ == "__main__":
  main()
