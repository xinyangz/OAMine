"""Run evaluation comparing a clustering result with ground truth labels"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

import plac
try:
  import wandb
  wandb_available = True
except:
  logger.warning("wandb not available")
  wandb_available = False

import eval
import utils




def to_clusters(phrases, labels, skip_noise=True):
  clusters = defaultdict(list)
  for phrase, label in zip(phrases, labels):
    if label == -1 and skip_noise:
      continue
    clusters[label].append(phrase)
  clusters = dict(clusters)
  return clusters


def load_eval_clusters(eval_file, topk=None):
  eval_docs = utils.JsonL.load(eval_file)
  eval_clusters = {}
  for eval_doc in eval_docs:
    attribute_type = eval_doc[0]
    attribute_values = eval_doc[1]
    if topk:
      attribute_values = attribute_values[:topk]
    eval_clusters[attribute_type] = attribute_values
  return eval_clusters


def count_in_vocab(eval_clusters: Dict[str, List[str]], phrases: List[str]):
  eval_phrases = {p for clus in eval_clusters.values() for p in clus}
  phrases = set(phrases)
  return len(eval_phrases & phrases)


def main(pred_dir=Path("/path/to/prediction"),
         eval_dir=Path("/path/to/output"),
         wandb_notes=None,
         use_wandb=False):
  print("Use wandb", use_wandb)
  pred_glob = "*.pred.jsonl"
  if use_wandb:
    wandb.init(notes=wandb_notes)
    wandb.config.pred_dir = pred_dir
  # wandb.init()
  purities = []
  jaccards = []
  nmis = []
  aris = []
  recalls = []

  for pred_file in Path(pred_dir).glob(pred_glob):
    pt_name = pred_file.stem[:-(len(pred_glob) - 2 - len(pred_glob.split(".")[-1]))]
    try:
      eval_clusters = load_eval_clusters(Path(eval_dir, f"{pt_name}.eval.jsonl"))
    except FileNotFoundError:
      continue

    n_eval_clusters = len(eval_clusters)
    if n_eval_clusters == 0:
      logger.warning(f"Eval file for {pt_name} shows 0 clusters")
      continue

    pred_clusters = load_eval_clusters(pred_file, topk=8000)

    preds = dict()
    all_pred_phrases = set()
    for i, (pred_label, pred_phrases) in enumerate(pred_clusters.items()):
      for p in pred_phrases:
        preds[p] = i
        all_pred_phrases.add(p)

    all_eval_phrases = set()
    for eval_phrases in eval_clusters.values():
      all_eval_phrases |= set(eval_phrases)

    try:
      jaccard, n_used = eval.set_eval_jaccard(preds, eval_clusters, return_in_vocab=True)
    except ZeroDivisionError:
      jaccard = 0
    try:
      purity = eval.set_eval_purity(preds, eval_clusters)
    except ZeroDivisionError:
      purity = 0
    try:
      nmi = eval.set_eval_nmi(preds, eval_clusters)
    except ZeroDivisionError:
      nmi = 0
    try:
      ari = eval.set_eval_ari(preds, eval_clusters)
    except ZeroDivisionError:
      ari = 0
    try:
      recall = eval.set_eval_recall(preds, eval_clusters)
    except ZeroDivisionError:
      recall = 0


    purities.append(purity)
    jaccards.append(jaccard)
    recalls.append(recall)
    nmis.append(nmi)
    aris.append(ari)
  avg_purity = sum(purities) / len(purities)
  avg_jaccard = sum(jaccards) / len(jaccards)
  avg_nmi = sum(nmis) / len(nmis)
  avg_ari = sum(aris) / len(aris)
  avg_recall = sum(recalls) / len(recalls)

  print(f"{len(purities)} valid PT evals")
  print(f"avg purity {avg_purity:.4f}")
  print(f"avg jaccard {avg_jaccard:.4f}")
  print(f"avg nmi: {avg_nmi:.4f}")
  print(f"avg ari: {avg_ari:.4f}")
  print(f"avg recall {avg_recall:.4f}")

  print(f"{avg_ari}\t{avg_jaccard}\t{avg_nmi}\t{avg_recall}")

  if use_wandb and wandb_available:
    wandb.summary.avg_purity = avg_purity
    wandb.summary.avg_jaccard = avg_jaccard
    wandb.summary.avg_recall = avg_recall
    wandb.summary.avg_nmi = avg_nmi
    wandb.summary.avg_ari = avg_ari
    wandb.summary.n_valid_pt = len(purities)



if __name__ == "__main__":
  plac.call(main)



