"""Sequential self-ensemble inference based on embedding and classifier results."""

import argparse
import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from sklearn.cluster import DBSCAN

import utils

logger = logging.getLogger()

Cluster = List[str]
ClusterName = str
Clusters = Dict[ClusterName, Cluster]

DBSCAN_JOBS = 16  # hard coded, whatever


def get_product_types(exp_dir: Path):
  pts = []
  for pt_file in Path(exp_dir).glob("*.emb.pkl"):
    pt_name = pt_file.stem[:-len(".emb")]
    pts.append(pt_name)

  if not pts:
    raise RuntimeError("No embedding file found from exp_dir! Run embedding first.")
  else:
    logger.warning(f"{len(pts)} PT embedding files found!")
    logger.info(f"First 3 PTs: {pts[:3]}")
  return pts


def to_clusters(phrases: List[str], labels: List[Union[int, str]], skip_noise=True) -> Clusters:
  """Convert aligned phrases and their labels to dictionary cluster format."""
  clusters = defaultdict(list)
  for phrase, label in zip(phrases, labels):
    if label == -1 and skip_noise:
      continue
    clusters[str(label)].append(phrase)
  # clusters = list(clusters.values())
  # clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
  clusters = dict(clusters)
  return clusters


def get_phrase_set_from_clusters(clusters: Clusters):
  phrase_set = set()
  for phrases in clusters.values():
    phrase_set |= set(phrases)
  return phrase_set


def match_clusters(dbscan_clusters, clf_file):
  """Match classifier generated clusters with DBSCAN clusters.

  Returns:
    clf2dbscan: a dictionary mapping clf cluster number to dbscan cluster number
    clf_clusters: a dictionary mapping clf cluster number to list of phrases in that cluster"""
  clf2dbscan = dict()
  clf_cluster2phrases = defaultdict(list)
  with clf_file.open("r") as f:
    # format of the file
    # phrase \t cluster_name
    for line in f:
      phrase, clf_label = line.strip().split("\t")
      clf_cluster2phrases[clf_label].append(phrase)
  # try alignment
  for clf_cluster, clf_phrases in clf_cluster2phrases.items():
    clf_phrases = set(clf_phrases)
    dbscan_labels = []
    match_ratios = []
    for dbscan_cluster, dbscan_phrases in dbscan_clusters.items():
      n_common = len(set(dbscan_phrases) & clf_phrases)
      match_ratio = n_common / len(clf_phrases) if len(clf_phrases) > 0 else 0
      match_ratios.append(match_ratio)
      dbscan_labels.append(dbscan_cluster)
    best_match = np.argmax(match_ratios)
    clf2dbscan[clf_cluster] = dbscan_labels[best_match]
  clf_clusters = dict(clf_cluster2phrases)
  return clf2dbscan, clf_clusters


def merge_clusters(dbscan_clusters: Clusters, clf_clusters: Clusters, clf2dbscan: Dict[str, str]) -> Clusters:
  """Merge dbscan and classifier clusters.

  Try to put classifier predictions into dbscan predictions, if they do not
  exist in those predictions already.

  Args:
    dbscan_clusters: a dictionary of dbscan clusters
    clf_clusters: a dictionary of classifier clusters
    clf2dbscan: a dictionary mapping classifier cluster labels to dbscans'

  Returns:
    a dictionary of merged clusters"""
  existing_preds = get_phrase_set_from_clusters(dbscan_clusters)

  merged_clusters = copy.deepcopy(dbscan_clusters)

  n_merged = 0
  for clf_cluster, clf_phrases in clf_clusters.items():
    dbscan_label = clf2dbscan[clf_cluster]
    for phrase in clf_phrases:
      if phrase not in existing_preds:
        merged_clusters[dbscan_label].append(phrase)
        existing_preds.add(phrase)
        n_merged += 1

  logger.info(f"Merged {n_merged} phrases from clf prediction to dbscan")
  # print((f"Merged {n_merged} phrases from clf prediction to dbscan"))
  return merged_clusters


def run_mixed_inference_pt(config):
  cfg = config["cfg"]
  pt = config["pt"]

  output_path = Path(cfg.output_dir)
  utils.IO.ensure_dir(output_path)

  emb_file = Path(cfg.emb_dir, f"{pt}.emb.pkl")
  if not emb_file:
    raise FileNotFoundError(f"{emb_file} not found, must run embedding first")

  use_clf = False
  if cfg.clf_dir is not None:
    use_clf = True

  emb: Dict[str, np.ndarray] = utils.Pkl.load(emb_file)
  vecs = []
  phrases = []
  for phrase, vec in emb.items():
    phrases.append(phrase)
    vecs.append(vec)
  vecs = np.vstack(vecs)

  # run DBSCAN inference
  clus = DBSCAN(eps=cfg.dbscan_thres, min_samples=cfg.dbscan_min_pts, metric="cosine", n_jobs=DBSCAN_JOBS)
  clus.fit_predict(vecs)

  # preds = clus.labels_
  # preds = {phrase: label for phrase, label in zip(phrases, clus.labels_) if label != -1}
  dbscan_clusters = to_clusters(phrases, clus.labels_, skip_noise=True)

  # Add classifier results w/ self-ensemble
  if use_clf:
    logger.info(f"[{pt}] Trying to add classifier predictions")
    clf_file = Path(cfg.clf_dir, f"{pt}.pred.txt")
    if clf_file.is_file():
      # step: 1: match classifier clusters to dbscan clusters
      clf2dbscan, clf_clusters = match_clusters(dbscan_clusters, clf_file)

      # step 2: merge clf and dbscan predictions
      merged_clusters = merge_clusters(dbscan_clusters, clf_clusters, clf2dbscan)
    else:
      logger.warning(f"[{pt}] CLF inference file not found, use dbscan only for {pt}")
      merged_clusters = dbscan_clusters
  else:
    logger.info("[{pt}] Using only DBSCAN predictions")
    merged_clusters = dbscan_clusters

  # save predictions to a new file
  clus_output_file = Path(output_path, f"{pt}.pred.jsonl")
  output_docs = []
  for i, clus in merged_clusters.items():
    output_docs.append((f"C_{i}", clus))
  utils.JsonL.save(clus_output_file, output_docs)

  # save noise to another file
  pred_phrase_set = get_phrase_set_from_clusters(merged_clusters)
  noise = set(phrases) - pred_phrase_set
  # print(f"{pt} non noise {len(pred_phrase_set)}")
  noise_cluster = list(noise)
  noise_output_file = Path(output_path, f"{pt}.noise.txt")
  utils.TextIO.save(noise_output_file, noise_cluster)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # data args
  parser.add_argument("--exp_dir", help="Path to experiment (embedding output)", default=None, type=Path)
  parser.add_argument("--output_name",
                      help="Name for output directory. Will create under exp_dir if doesn't exist",
                      default="iter_1",
                      type=str)

  # dbscan args
  parser.add_argument("--dbscan_thres", default=0.05, type=float)
  parser.add_argument("--dbscan_min_pts", default=5, type=int)

  # inference option
  parser.add_argument("--use_clf", help="Flag to use self-ensemble of classifier and DBSCAN", action="store_true")


  args = parser.parse_args()

  utils.Log.config_logging()

  # get all PTs that has embedding file generated
  pts = get_product_types(args.exp_dir)
  # # DEBUG
  # pts = pts[:3]

  # run sequential inference
  from tqdm import tqdm
  for pt in tqdm(pts):
    config = {
      "args": args,
      "pt": pt
    }
    run_mixed_inference_pt(config)
