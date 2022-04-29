"""Generate classifier training data from seeds (predictions)
See config/preprocessing/clf for configurations."""

import sys
from pathlib import Path

sys.path.insert(1, Path(sys.path[0]).parent.as_posix())  # to include parent dir imports

import itertools
import logging
import random
import re
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import hydra
import utils
from flashtext import KeywordProcessor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

EntityContext = namedtuple("EntityContext", ["left_context", "entity", "right_context"])
ContextualizedExample = namedtuple("ContextualizedExample", ["entities", "label"])


def example_to_dict(example):
  phrase_context, pt_name, label = example
  ret = {
      "left_context": phrase_context.left_context,
      "entity": phrase_context.entity,
      "right_context": phrase_context.right_context,
      "pt_name": pt_name,
      "label": label,
  }
  return ret


def read_seeds_from_file(fname) -> Tuple[List[List[str]], List[str]]:
  # read seeds
  seed_docs: List[str, List[str]] = utils.JsonL.load(fname)
  # convert to seed clusters
  names = []
  seeds: List[List[str]] = []
  for attribute, attribute_values in seed_docs:
    seeds.append(attribute_values)
    names.append(attribute)
  return seeds, names


def generate_from_seeds(seeds: List[List[str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
  # generate positive and negative pairs
  positive_pairs = []
  for seed_values in seeds:
    for pair in itertools.combinations(seed_values, 2):
      positive_pairs.append(pair)

  negative_pairs = []
  for i in range(len(seeds)):
    seed_values_a = seeds[i]
    for j in range(i + 1, len(seeds)):
      seed_values_b = seeds[j]
      for sa in seed_values_a:
        for sb in seed_values_b:
          negative_pairs.append((sa, sb))
  return positive_pairs, negative_pairs


def generate_triplet_from_seeds(seeds: List[List[str]]) -> List[Tuple[str, str, str]]:
  # generate positive and negative pairs
  triplets = []
  all_words = {phrase for cluster in seeds for phrase in cluster}
  for i, seed_values in enumerate(seeds):
    other_words = list(all_words - set(seed_values))
    if len(other_words) == 0:
      continue
    for pair in itertools.combinations(seed_values, 2):
      if random.random() > 0.5:
        pair = (pair[1], pair[0])
      neg = random.choice(other_words)
      triplet = (*pair, neg)
      triplets.append(triplet)
  return triplets


def generate_from_asin_reg(docs: List[List[str]], sample_per_asin=3):
  negative_pairs = []
  for doc in docs:
    if len(doc) * (len(doc) - 1) / 2 < sample_per_asin:
      continue
    pairs = utils.Rnd.random_pairs(doc, sample_per_asin)
    for pair in pairs:
      negative_pairs.append(tuple(pair))
  return negative_pairs


def augment_triplet_from_asins(triplets: List[Tuple[str, str, str]], docs: List[List[str]], sample_per_asin=3):
  negative_profile = defaultdict(list)  # word: list of negative words
  for doc in docs:
    if len(doc) * (len(doc) - 1) / 2 < sample_per_asin:
      continue
    pairs = utils.Rnd.random_pairs(doc, sample_per_asin)
    for pair in pairs:
      negative_profile[pair[0]].append(pair[1])
      negative_profile[pair[1]].append(pair[0])

  positive_profile = defaultdict(set)
  for anchor, pos, _ in triplets:
    positive_profile[anchor].add(pos)
    positive_profile[pos].add(anchor)

  augmented_triplets = []
  for anchor, pos, _ in triplets:
    if anchor in negative_profile:
      neg_asin = random.choice(negative_profile[anchor])
      if neg_asin not in positive_profile[anchor]:
        augmented_triplets.append((anchor, pos, neg_asin))
    if pos in negative_profile:
      neg_asin = random.choice(negative_profile[pos])
      if neg_asin not in positive_profile[pos]:
        augmented_triplets.append((pos, anchor, neg_asin))
  return augmented_triplets


def sample_positive_negative(positive_pairs,
                             negative_pairs,
                             negative_pairs_asin,
                             times_negative=3,
                             times_asin_negative=5):
  """Given all pairs. Sample:
    1. all positive pairs
    2. (times_negative * #positive) negative pairs from seeds
    3. (times_asin_negative * #negative) negative pairs from ASIN regularization"""
  # organize pairs from seeds and ASINs into dataset
  n_neg_asked = len(positive_pairs) * times_negative
  n_neg_from_seeds = min(n_neg_asked, len(negative_pairs))
  if n_neg_from_seeds < n_neg_asked:
    logger.warning(f"Asked for {n_neg_asked} negative pairs from seeds, but only have {len(negative_pairs)}")
  sampled_neg_seed = random.sample(negative_pairs, n_neg_from_seeds)

  n_neg_asins_asked = n_neg_from_seeds * times_asin_negative
  n_neg_asins = min(n_neg_asins_asked, len(negative_pairs_asin))
  if n_neg_asins < n_neg_asins_asked:
    logger.warning(f"Asked for {n_neg_asins_asked} negative pairs from ASINs, but only have {len(negative_pairs)}")
  sampled_neg_asins = random.sample(negative_pairs_asin, n_neg_asins)

  random.shuffle(positive_pairs)
  random.shuffle(sampled_neg_asins)
  random.shuffle(sampled_neg_seed)

  return positive_pairs, sampled_neg_seed, sampled_neg_asins


def save_datasets_clf(ds_train, ds_dev, output_dir):
  utils.IO.ensure_dir(output_dir)
  output_file = Path(output_dir, f"train.jsonl")
  logger.warning(f"Saving to {output_file}")
  output_objs = []
  for example in ds_train:
    output_objs.append(example_to_dict(example))

  utils.JsonL.save(output_file, output_objs)
  logger.info(f"Train set size {len(output_objs)}")

  output_file = Path(output_dir, f"dev.jsonl")
  logger.warning(f"Saving to {output_file}")
  output_objs = []
  for example in ds_dev:
    output_objs.append(example_to_dict(example))
  utils.JsonL.save(output_file, output_objs)
  logger.info(f"Dev set size {len(output_objs)}")


def match_phrase2context(phrases, docs):
  raw_texts = [" ".join(doc) for doc in docs]

  # string matching
  phrase2context: Dict[str, List[EntityContext]] = defaultdict(list)
  kw_processor = KeywordProcessor()
  kw_processor.add_keywords_from_list(phrases)
  for raw_text in raw_texts:
    keywords_found = kw_processor.extract_keywords(raw_text, span_info=True)
    for kw, start, end in keywords_found:
      left_ctx = raw_text[:start].strip()
      right_ctx = raw_text[end:].strip()
      phrase2context[kw].append(EntityContext(left_ctx, kw, right_ctx))
  phrase2context = dict(phrase2context)
  return phrase2context


def match_context(ds: List[Tuple[str, str, int]],
                  docs: List[List[str]],
                  sample_context=2) -> List[ContextualizedExample]:
  """Args:
    - ds: list of tuples, context-free dataset
    - docs: list of candidate generation outputs
    - sample_context: number of contextualized sampled generated from each context-free phrase"""
  phrases = []
  for tuple in ds:
    p1 = tuple[0]
    p2 = tuple[1]
    phrases.append(p1)
    phrases.append(p2)
  phrases = list(set(phrases))

  phrase2context = match_phrase2context(phrases, docs)

  # for each pair, try to find contextualized examples
  contextualized_ds = []

  def _sample_contextualized(phrase, contexts, k):
    if len(contexts) == 0:
      return [EntityContext("", phrase, "")]
    elif len(contexts) < k:
      return contexts
    else:
      return random.sample(contexts, k)

  for p1, p2, label in ds:
    p1_context = phrase2context.get(p1, [])
    p2_context = phrase2context.get(p2, [])

    p1_samples = _sample_contextualized(p1, p1_context, sample_context)
    p2_samples = _sample_contextualized(p2, p2_context, sample_context)

    for p1s in p1_samples:
      for p2s in p2_samples:
        contextualized_ds.append(ContextualizedExample([p1s, p2s], label))
  return contextualized_ds


def make_dataset_clf(pt: str,
                     seeds: List[List[str]],
                     docs: List[List[str]],
                     min_examples_per_set=0,
                     max_examples_per_set=None,
                     n_ctx_sample=3,
                     pct_dev=0) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, int]]]:
  ds_train = []
  ds_dev = []

  # convert to tuple format
  examples = []
  labels = []
  phrases = set()
  for i, seed_values in enumerate(seeds):
    if len(seed_values) < min_examples_per_set:
      logger.info(f"[{pt}] Too few examples in class {i}, skipping")
      continue
    if max_examples_per_set is not None and len(seed_values) > max_examples_per_set:
      logger.debug("Subsampling")
      seed_values = random.sample(seed_values, max_examples_per_set)
    for seed_value in seed_values:
      examples.append((seed_value, pt, f"{pt}_C_{i}"))
      labels.append(f"{pt}_C_{i}")
      phrases.add(seed_value)
  phrases = list(phrases)

  # match context
  phrase2context = match_phrase2context(phrases, docs)

  n_split = int(len(examples) * pct_dev)

  def _sample_context(phrase, phrase2context, k):
    contexts = phrase2context.get(phrase, [])
    if len(contexts) == 0:
      return [EntityContext("", phrase, "")]
    elif len(contexts) < k:
      return contexts
    else:
      return random.sample(contexts, k)

  # stratified sampling
  if len(examples) == 0:
    logger.warning(f"0 examples found for pt={pt}")
    return [], []
  if pct_dev > 0:
    assert pct_dev < 1
    train_examples, dev_examples = train_test_split(examples, test_size=pct_dev, stratify=labels)
  else:
    train_examples = examples
    dev_examples = []

  for phrase, pt_name, label in dev_examples:  # dev
    for phrase_context in _sample_context(phrase, phrase2context, n_ctx_sample):
      ds_dev.append((phrase_context, pt_name, label))

  for phrase, pt_name, label in train_examples:  # train
    for phrase_context in _sample_context(phrase, phrase2context, n_ctx_sample):
      ds_train.append((phrase_context, pt_name, label))

  # n_total = len(ds_train) + len(ds_dev)
  # logger.info(f"{n_total} pairs in dataset, {len(pos_pairs)} positive, {n_total - len(pos_pairs)} negative, {len(neg_pairs)} neg from seeds, {len(neg_pairs_asin)} neg from ASINs")
  # logger.info(f"{len(ds_train)} train, {len(ds_dev)} dev")

  return ds_train, ds_dev


def split_by_attr_random(pt2seeds: Dict[str, List[List[str]]],
                         pt2seed_names: Dict[str, List[str]],
                         candidate_dir: Path,
                         output_dir: Path,
                         min_examples_per_set=0,
                         max_examples_per_set=None,
                         n_ctx_sample=10,
                         pct_dev=0.2):
  ds_train = []
  ds_dev = []

  for pt, seeds in tqdm(pt2seeds.items(), desc="Sample clf examples"):
    logger.debug(pt)
    docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))

    ds_train_pt, ds_dev_pt = make_dataset_clf(pt,
                                              seeds,
                                              docs,
                                              min_examples_per_set=min_examples_per_set,
                                              max_examples_per_set=max_examples_per_set,
                                              n_ctx_sample=n_ctx_sample,
                                              pct_dev=pct_dev)

    ds_train.extend(ds_train_pt)
    ds_dev.extend(ds_dev_pt)

  save_datasets_clf(ds_train, ds_dev, output_dir)


@hydra.main(config_path="../../exp_config", config_name="config")
def main(global_cfg):

  cfg = global_cfg.preprocessing.clf
  random.seed(cfg.rnd_seed)

  logger.info("Data generation for classification objective")

  seed_dir = Path(cfg.seed_dir)
  candidate_dir = Path(cfg.candidate_dir)
  if not cfg.output_dir:
    raise RuntimeError("Output dir must be set")
  output_dir = Path(cfg.output_dir)
  if not candidate_dir.is_dir():
    raise RuntimeError("Must run candidate generation before generating data.")
  if not output_dir.is_dir():
    logger.warning(f"Creating output dir {output_dir}")
    output_dir.mkdir(parents=True)
  elif cfg.overwrite_output:
    logger.warning(f"Overwriting output dir {output_dir}")
  else:
    raise RuntimeError("output_dir exists. To overwrite, pass run.overwrite_output=true.")

  pt2seeds = dict()
  pt2seed_names = dict()
  if cfg.inference_dir is not None:
    logger.info("inference_dir set, using it for data generation")
    # iterative training
    # generate data by using predictions from last iteration as seed
    inference_dir = Path(cfg.inference_dir)
    for seed_file in inference_dir.glob("*.pred.jsonl"):
      pt_name = seed_file.stem[:-len(".seed")]
      seeds, names = read_seeds_from_file(seed_file)
      pt2seeds[pt_name] = seeds
      pt2seed_names[pt_name] = names
  else:
    logger.info("Using seed values for data generation")
    # first iteration
    # use input seed values to generate data
    for seed_file in seed_dir.glob("*.seed.jsonl"):
      pt_name = re.match(r"(.*).seed", seed_file.stem)[1]
      seeds, names = read_seeds_from_file(seed_file)
      pt2seeds[pt_name] = seeds
      pt2seed_names[pt_name] = names

  # split 1 (PT level generalization): hold out several PTs for evaluation
  # split_by_pt(pt2seeds, pt2seed_names, candidate_dir, output_dir=Path(data_dir, "pt_split"))

  # split 2 (attribute level generalization): hold out 1 attribute from each PT for evaluation (if there are more than 3 attributes)
  # split_by_attr(pt2seeds, pt2seed_names, candidate_dir, output_dir=Path(data_dir, "attr_split"))

  # split 3 (easy case): for each pt, hold out randomly selected pairs for evaluation
  split_by_attr_random(pt2seeds,
                       pt2seed_names,
                       candidate_dir,
                       min_examples_per_set=cfg.min_examples_per_set,
                       max_examples_per_set=cfg.max_examples_per_set,
                       n_ctx_sample=cfg.contexts_per_sample,
                       output_dir=cfg.output_dir,
                       pct_dev=cfg.pct_dev)


if __name__ == "__main__":
  main()
