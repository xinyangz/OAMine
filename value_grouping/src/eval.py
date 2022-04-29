import itertools
import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, adjusted_rand_score

logger = logging.getLogger(__name__)

def clustering_perm_acc(pred_labels, true_labels):
  """Align clusters to ground truth and compute accuracy"""
  true_label_set = list(set(true_labels))
  logger.info(f"# of unique true labels {len(true_label_set)}")
  mapping = {label: i for i, label in enumerate(true_label_set)}
  true_labels = [mapping[l] for l in true_labels]

  # find all permutations
  best_acc = 0
  all_labels = list(range(len(true_label_set)))
  for perm in itertools.permutations(all_labels):
    perm_mapping = {p: i for i, p in enumerate(perm)}
    pred_labels_mapped = [perm_mapping[l] for l in pred_labels]
    acc = accuracy_score(true_labels, pred_labels_mapped)
    best_acc = max(acc, best_acc)
  return best_acc


def greedy_match_acc(pred_labels, true_labels):
  label2instances = defaultdict(list)
  for i, true_label in enumerate(true_labels):
    label2instances[true_label].append(i)
  label2instances = {lb: frozenset(instances) for lb, instances in label2instances.items()}

  all_pred_labels = set(pred_labels)
  true2pred_map = dict()
  for lb in all_pred_labels:
    if len(true2pred_map) > len(label2instances):
      logger.warning("Number of predicted clusters more than number of true clusters")
      break
    instances = {i for i, l in enumerate(pred_labels) if l == lb}
    matches = []
    for true_lb, true_instances in label2instances.items():
      if true_lb in true2pred_map:
        continue
      matches.append((true_lb, len(true_instances & instances)))
    matches = list(sorted(matches, key=lambda x: x[1], reverse=True))
    matched_label = matches[0][0]
    true2pred_map[matched_label] = lb
  if len(true2pred_map) < len(label2instances):
    logger.warning("Number of predicted clusters less than number of true clusters")
  pred2true_map = {v: k for k, v in true2pred_map.items()}
  pred_mapped = [pred2true_map[p] for p in pred_labels]
  return accuracy_score(true_labels, pred_mapped)


def classification_acc(pred_labels, true_labels):
  return accuracy_score(true_labels, pred_labels)


def jaccard(pred_label_pairs: List[Tuple[int, int]]):
  pl_count = defaultdict(int)
  p_count = defaultdict(int)
  l_count = defaultdict(int)
  n_total = len(pred_label_pairs)

  for pred, label in pred_label_pairs:
    p_count[pred] += 1
    l_count[label] += 1
    pl_count[(pred, label)] += 1

  n_pred = len(p_count)  # num of predicted clusters
  n_label = len(l_count)

  TP = 0
  FN = 0
  FP = 0
  for (p, l), count in pl_count.items():
    TP += count * (count - 1) / 2

  for p, count in p_count.items():
    FN += count * (count - 1) / 2
  FN -= TP

  for l, count in l_count.items():
    FP += count * (count - 1) / 2
  FP -= TP

  return TP / (TP + FN + FP)


def purity(pred_label_pairs: List[Tuple[int, int]], debug=False):
  pl_count = defaultdict(int)
  p_count = defaultdict(int)
  l_count = defaultdict(int)

  for pred, label in pred_label_pairs:
    p_count[pred] += 1
    l_count[label] += 1
    pl_count[(pred, label)] += 1

  all_max = []
  all_count = 0
  for p, count in p_count.items():
    # for each predicted cluster
    # find largest subset that belongs to the same true cluster
    p_max = 0
    match_cluster = 0
    for l in l_count:
      if (p, l) not in pl_count:
        continue
      if pl_count[(p, l)] > p_max:
        p_max = pl_count[(p, l)]
        match_cluster = l
    all_max.append(p_max)
    all_count += count
    if debug:
      print("pred cluster", p, "pred count", count, "max match", p_max, "match cluster", match_cluster)

  return sum(all_max) / all_count


def nmi(pred_label_pairs: List[Tuple[int, int]], debug=False):
  pl_count = defaultdict(int)
  p_count = defaultdict(int)
  l_count = defaultdict(int)
  n_total = len(pred_label_pairs)

  for pred, label in pred_label_pairs:
    p_count[pred] += 1
    l_count[label] += 1
    pl_count[(pred, label)] += 1

  # compute TP, I_c_t
  I_p_l = 0
  for (pred, label), count in pl_count.items():
      p_pred_label = count / n_total
      p_pred = p_count[pred] / n_total
      p_label = l_count[label] / n_total
      I_p_l += p_pred_label * math.log2(p_pred_label / p_pred / p_label)


  # compute FN, FP, H_C, H_T
  H_pred = 0
  H_label = 0
  for pred, count in p_count.items():
      p_pred = count / n_total
      H_pred += p_pred * math.log2(p_pred)

  for label, count in l_count.items():
      p_label = count / n_total
      H_label += p_label * math.log2(p_label)

  # generate output
  nmi = I_p_l / math.sqrt(H_pred * H_label)
  return nmi


# def b_cubed(pred_label_pairs: List[Tuple[int, int]], debug=False):
#   pl_count = defaultdict(int)
#   p_count = defaultdict(int)
#   l_count = defaultdict(int)
#   n_total = len(pred_label_pairs)

#   for pred, label in pred_label_pairs:
#     p_count[pred] += 1
#     l_count[label] += 1
#     pl_count[(pred, label)] += 1

#   # precisions, recalls, f1s
#   f1s = []
#   for (pred, label), count in pl_count.items():
#     n_p = pl_count


#   # compute FN, FP, H_C, H_T
#   H_pred = 0
#   H_label = 0
#   for pred, count in p_count.items():
#       p_pred = count / n_total
#       H_pred += p_pred * math.log2(p_pred)

#   for label, count in l_count.items():
#       p_label = count / n_total
#       H_label += p_label * math.log2(p_label)

#   # generate output
#   nmi = I_p_l / math.sqrt(H_pred * H_label)
#   return nmi

def matching_tp(pred_label_pairs: List[Tuple[int, int]], debug=False):
  pl_count = defaultdict(int)
  p_count = defaultdict(int)
  l_count = defaultdict(int)

  for pred, label in pred_label_pairs:
    p_count[pred] += 1
    l_count[label] += 1
    pl_count[(pred, label)] += 1

  TP = 0
  for p, count in p_count.items():
    # for each predicted cluster
    # find largest subset that belongs to the same true cluster
    p_max = 0
    match_cluster = 0
    for l in l_count:
      if (p, l) not in pl_count:
        continue
      if pl_count[(p, l)] > p_max:
        p_max = pl_count[(p, l)]
    TP += p_max

  return TP


def get_pred_label_pairs(preds, labels, verbose=False, debug=False):
  pred_label_pairs = []
  n_total = 0
  n_total_invocab = 0
  for i, (attribute_type, attribute_values) in enumerate(labels.items()):
    n_total_at = 0
    n_total_invocab_at = 0
    for av in attribute_values:
      n_total += 1
      n_total_at += 1
      if av in preds:
        pred_label_pairs.append((preds[av], i))
        n_total_invocab += 1
        n_total_invocab_at += 1
    if verbose:
      logger.info(f"AT = {attribute_type}, {n_total_invocab_at} out of {n_total_at} in vocab")
    if debug:
      print(i, attribute_type)
      print(n_total_invocab_at, "out of", n_total_at, "eval values in vocab")
  if verbose:
    logger.info(f"Overall, {n_total_invocab} out of {n_total} in vocab")
  return pred_label_pairs


def set_eval_jaccard(preds: Dict[str, int], labels: Dict[str, List[str]], verbose=False, return_in_vocab=False):
  pred_label_pairs = get_pred_label_pairs(preds, labels, verbose)
  res = 0 if len(pred_label_pairs) == 0 else jaccard(pred_label_pairs)
  pred_phrases = set(preds.keys())
  eval_phrases = {p for clus in labels.values() for p in clus}
  in_vocab = len(pred_phrases & eval_phrases)
  if return_in_vocab:
    return res, in_vocab
  else:
    return res


def set_eval_nmi(preds: Dict[str, int], labels: Dict[str, List[str]], verbose=False, return_in_vocab=False):
  pred_label_pairs = get_pred_label_pairs(preds, labels, verbose)
  res = 0 if len(pred_label_pairs) == 0 else nmi(pred_label_pairs)
  pred_phrases = set(preds.keys())
  eval_phrases = {p for clus in labels.values() for p in clus}
  in_vocab = len(pred_phrases & eval_phrases)
  if return_in_vocab:
    return res, in_vocab
  else:
    return res


def set_eval_recall(preds: Dict[str, int], labels: Dict[str, List[str]], verbose=False, return_in_vocab=False):
  pred_label_pairs = get_pred_label_pairs(preds, labels, verbose)
  TP = 0 if len(pred_label_pairs) == 0 else matching_tp(pred_label_pairs)
  eval_phrases = {p for clus in labels.values() for p in clus}
  return TP / len(eval_phrases)


def set_eval_ari(preds: Dict[str, int], labels: Dict[str, List[str]], verbose=False, return_in_vocab=False):
  pred_label_pairs = get_pred_label_pairs(preds, labels, verbose)
  # res = 0 if len(pred_label_pairs) == 0 else jaccard(pred_label_pairs)
  if len(pred_label_pairs) == 0:
    return 0
  preds, labels = list(zip(*pred_label_pairs))
  return adjusted_rand_score(labels, preds)


def set_eval_purity(preds: Dict[str, int], labels: Dict[str, List[str]], verbose=False, return_in_vocab=False):
  pred_label_pairs = get_pred_label_pairs(preds, labels, verbose)
  res = 0 if len(pred_label_pairs) == 0 else purity(pred_label_pairs)
  pred_phrases = set(preds.keys())
  eval_phrases = {p for clus in labels.values() for p in clus}
  in_vocab = len(pred_phrases & eval_phrases)
  if return_in_vocab:
    return res, in_vocab
  else:
    return res


def set_eval_one_out(eval_fn, preds, labels):
  cluster_names = set(preds.values())
  best_performance = -1
  phrases_at_best = set()
  for clus in cluster_names:
    preds_one_out = {k: v for k, v in preds.items() if v != clus}
    phrase_set = set(preds_one_out.keys())
    performance = eval_fn(preds_one_out, labels)
    if performance > best_performance:
      best_performance = performance
      phrases_at_best = phrase_set
  return best_performance, phrases_at_best
