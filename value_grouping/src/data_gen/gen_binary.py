"""Generate binary training data from seeds (predictions) and unlabeled
See config/preprocessing/binary for configurations."""
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
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

EntityContext = namedtuple("EntityContext", ["left_context", "entity", "right_context"])
ContextualizedExample = namedtuple("ContextualizedExample", ["entities", "label"])


def example_to_dict(example):
  ent_a, ent_b = example.entities
  label = example.label
  ret = {
      "entity_a": {
          "left_context": ent_a.left_context,
          "entity": ent_a.entity,
          "right_context": ent_a.right_context,
      },
      "entity_b": {
          "left_context": ent_b.left_context,
          "entity": ent_b.entity,
          "right_context": ent_b.right_context,
      },
      "label": label
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


def generate_from_seeds(seeds: List[List[str]],
                        max_pos_pairs_per_set=None) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
  # generate positive and negative pairs
  positive_pairs = []
  for seed_values in seeds:
    if (max_pos_pairs_per_set is not None) and len(seed_values) * (len(seed_values) - 1) / 2 > max_pos_pairs_per_set:
      logger.debug("Subsampling positive pairs from seed set")
      positive_pairs = utils.Rnd.random_pairs(seed_values, max_pos_pairs_per_set)
    else:
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


def generate_from_asin_reg(docs: List[List[str]], sample_per_asin=3):
  negative_pairs = []
  for doc in docs:
    if len(doc) * (len(doc) - 1) / 2 < sample_per_asin:
      continue
    pairs = utils.Rnd.random_pairs(doc, sample_per_asin)
    for pair in pairs:
      negative_pairs.append(tuple(pair))
  return negative_pairs


def sample_positive_negative(positive_pairs,
                             negative_pairs,
                             negative_pairs_asin,
                             pos_cap=None,
                             times_negative=3,
                             times_asin_negative=5):
  """Given all pairs. Sample:
    1. all positive pairs
    2. (times_negative * #positive) negative pairs from seeds
    3. (times_asin_negative * #negative) negative pairs from ASIN regularization"""
  if pos_cap and pos_cap < len(positive_pairs):
    positive_pairs = random.sample(positive_pairs, pos_cap)
  logger.debug(f"#pos_pairs {len(positive_pairs)}")
  # organize pairs from seeds and ASINs into dataset
  n_neg_asked = len(positive_pairs) * times_negative
  n_neg_from_seeds = min(n_neg_asked, len(negative_pairs))
  if n_neg_from_seeds < n_neg_asked:
    logger.warning(f"Asked for {n_neg_asked} negative pairs from seeds, but only have {len(negative_pairs)}")
  sampled_neg_seed = random.sample(negative_pairs, n_neg_from_seeds)

  n_neg_asins_asked = n_neg_from_seeds * times_asin_negative
  n_neg_asins = min(n_neg_asins_asked, len(negative_pairs_asin))
  if n_neg_asins < n_neg_asins_asked:
    logger.warning(f"Asked for {n_neg_asins_asked} negative pairs from ASINs, but only have {len(negative_pairs_asin)}")
  sampled_neg_asins = random.sample(negative_pairs_asin, n_neg_asins)

  random.shuffle(positive_pairs)
  random.shuffle(sampled_neg_asins)
  random.shuffle(sampled_neg_seed)

  return positive_pairs, sampled_neg_seed, sampled_neg_asins


def make_dataset(pos_pairs, neg_pairs, neg_pairs_asin, pct_dev=0):
  ds_train = []
  ds_dev = []

  def _add_to_ds(ds_train, ds_dev, array, label, pct_dev):
    split = int(len(array) * pct_dev)
    for pair in array[:split]:
      ds_dev.append((*pair, label))
    for pair in array[split:]:
      ds_train.append((*pair, label))

  _add_to_ds(ds_train, ds_dev, pos_pairs, 1, pct_dev)
  _add_to_ds(ds_train, ds_dev, neg_pairs, -1, pct_dev)
  _add_to_ds(ds_train, ds_dev, neg_pairs_asin, -1, pct_dev)

  n_total = len(ds_train) + len(ds_dev)
  # logger.info(f"{n_total} pairs in dataset, {len(pos_pairs)} positive, {n_total - len(pos_pairs)} negative, {len(neg_pairs)} neg from seeds, {len(neg_pairs_asin)} neg from ASINs")
  # logger.info(f"{len(ds_train)} train, {len(ds_dev)} dev")

  return ds_train, ds_dev


def save_datasets(ds_train: List[ContextualizedExample], ds_dev: List[ContextualizedExample], output_dir):
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


def split_by_pt(pt2seeds: Dict[str, List[List[str]]],
                pt2seed_names: Dict[str, List[str]],
                candidate_dir: Path,
                output_dir: Path,
                pct_dev=0.2):
  pt2seeds = list(pt2seeds.items())
  random.shuffle(pt2seeds)

  ds_train = []
  ds_dev = []

  info = {"train_pts": [], "dev_pts": []}

  n_split = int(len(pt2seeds) * pct_dev)
  for pt, seeds in pt2seeds[n_split:]:
    docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))
    pos_pairs, neg_pairs = generate_from_seeds(seeds)
    neg_pairs_asin = generate_from_asin_reg(docs, sample_per_asin=3)
    pos_pairs_, neg_pairs_, neg_pairs_asin_ = sample_positive_negative(pos_pairs,
                                                                       neg_pairs,
                                                                       neg_pairs_asin,
                                                                       times_negative=3,
                                                                       times_asin_negative=5)
    ds_train_pt, _ = make_dataset(pos_pairs_, neg_pairs_, neg_pairs_asin_, pct_dev=0.)
    ds_train.extend(ds_train_pt)
    info["train_pts"].append(pt)

  for pt, seeds in pt2seeds[:n_split]:
    docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))
    pos_pairs, neg_pairs = generate_from_seeds(seeds)
    neg_pairs_asin = generate_from_asin_reg(docs, sample_per_asin=3)
    pos_pairs_, neg_pairs_, neg_pairs_asin_ = sample_positive_negative(pos_pairs,
                                                                       neg_pairs,
                                                                       neg_pairs_asin,
                                                                       times_negative=3,
                                                                       times_asin_negative=5)
    ds_dev_pt, _ = make_dataset(pos_pairs_, neg_pairs_, neg_pairs_asin_, pct_dev=0.)
    ds_dev.extend(ds_dev_pt)
    info["dev_pts"].append(pt)

  save_datasets(ds_train, ds_dev, output_dir)
  utils.Json.save(Path(output_dir, "info.json"), info)


def split_by_attr(pt2seeds: Dict[str, List[List[str]]],
                  pt2seed_names: Dict[str, List[str]],
                  candidate_dir: Path,
                  output_dir: Path,
                  hold_out=2):
  ds_train = []
  ds_dev = []

  N_ENSURE_TRAIN = 3

  info = {"dev_pts": [], "pt_info": dict()}

  for pt, seeds in pt2seeds.items():
    # if there are more than (hold_out + N_ENSURE_TRAIN) attribute seeds, hold out several for dev
    seed_names = pt2seed_names[pt]
    if len(seeds) >= hold_out + N_ENSURE_TRAIN:
      indices = list(range(len(seeds)))
      random.shuffle(indices)
      train_seeds = [seeds[i] for i in indices[hold_out:]]
      dev_seeds = [seeds[i] for i in indices[:hold_out]]
      train_seed_names = [seed_names[i] for i in indices[hold_out:]]
      dev_seed_names = [seed_names[i] for i in indices[:hold_out]]

      # sample dev set
      pos_pairs_dev, neg_pairs_dev = generate_from_seeds(dev_seeds)
      # Not using neg pairs from asin reg in dev set
      pos_pairs_, neg_pairs_, _ = sample_positive_negative(pos_pairs_dev,
                                                           neg_pairs_dev, [],
                                                           times_negative=3,
                                                           times_asin_negative=0)
      ds_pt, _ = make_dataset(pos_pairs_, neg_pairs_, [], pct_dev=0.)
      ds_dev.extend(ds_pt)
      info["dev_pts"].append(pt)
    else:
      train_seeds = seeds
      train_seed_names = seed_names
      dev_seeds = []
      dev_seed_names = []

    docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))
    pos_pairs, neg_pairs = generate_from_seeds(train_seeds)
    neg_pairs_asin = generate_from_asin_reg(docs, sample_per_asin=3)
    pos_pairs_, neg_pairs_, neg_pairs_asin_ = sample_positive_negative(pos_pairs,
                                                                       neg_pairs,
                                                                       neg_pairs_asin,
                                                                       times_negative=3,
                                                                       times_asin_negative=5)
    ds_train_pt, _ = make_dataset(pos_pairs_, neg_pairs_, neg_pairs_asin_, pct_dev=0.)
    ds_train.extend(ds_train_pt)

    info["pt_info"][pt] = {"train_attributes": train_seed_names, "dev_attributes": dev_seed_names}

  save_datasets(ds_train, ds_dev, output_dir)
  utils.Json.save(Path(output_dir, "info.json"), info)


def match_context(ds: List[Tuple[str, str, int]],
                  docs: List[List[str]],
                  sample_context=2) -> List[ContextualizedExample]:
  """Args:
    - ds: list of tuples, context-free dataset
    - docs: list of candidate outputs
    - sample_context: number of contextualized sampled generated from each context-free phrase"""
  phrases = []
  for tuple in ds:
    p1 = tuple[0]
    p2 = tuple[1]
    phrases.append(p1)
    phrases.append(p2)
  phrases = list(set(phrases))

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


def split_by_attr_random(pt2seeds: Dict[str, List[List[str]]],
                         pt2seed_names: Dict[str, List[str]],
                         candidate_dir: Path,
                         output_dir: Path,
                         neg_only_pts=None,
                         pos_per_asin=5,
                         times_negative=3,
                         times_asin_negative=5,
                         context_per_sample=2,
                         max_pos_pairs_per_set=None,
                         pct_dev=0.2):
  logger.info("Generate by random split")
  logger.info(f"pct_dev = {pct_dev}")
  ds_train = []
  ds_dev = []

  for pt, seeds in tqdm(pt2seeds.items(), desc="Sample pairs"):
    logger.debug(pt)
    docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))
    pos_pairs, neg_pairs = generate_from_seeds(seeds, max_pos_pairs_per_set=max_pos_pairs_per_set)
    neg_pairs_asin = generate_from_asin_reg(docs, sample_per_asin=pos_per_asin)
    pos_pairs_, neg_pairs_, neg_pairs_asin_ = sample_positive_negative(pos_pairs,
                                                                       neg_pairs,
                                                                       neg_pairs_asin,
                                                                       times_negative=times_negative,
                                                                       times_asin_negative=times_asin_negative)
    ds_train_pt, ds_dev_pt = make_dataset(pos_pairs_, neg_pairs_, neg_pairs_asin_, pct_dev=pct_dev)

    ds_train_pt = match_context(ds_train_pt, docs, sample_context=context_per_sample)
    ds_dev_pt = match_context(ds_dev_pt, docs, sample_context=context_per_sample)

    ds_train.extend(ds_train_pt)
    ds_dev.extend(ds_dev_pt)

  if neg_only_pts:
    for pt in neg_only_pts:
      docs = utils.JsonL.load(Path(candidate_dir, f"{pt}.chunk.jsonl"))
      neg_pairs_asin = generate_from_asin_reg(docs, sample_per_asin=pos_per_asin)
      ds_train_pt, ds_dev_pt = make_dataset([], neg_pairs_asin, [], pct_dev=pct_dev)
      ds_train_pt = match_context(ds_train_pt, docs, sample_context=context_per_sample)
      ds_dev_pt = match_context(ds_dev_pt, docs, sample_context=context_per_sample)
      ds_train.extend(ds_train_pt)
      ds_dev.extend(ds_dev_pt)


  save_datasets(ds_train, ds_dev, output_dir)


@hydra.main(config_path="../../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.preprocessing.binary

  random.seed(cfg.rnd_seed)

  logger.info("Data generation for binary objective")

  seed_dir = Path(cfg.seed_dir)
  candidate_dir = Path(cfg.candidate_dir)
  if cfg.neg_only_pts:
    neg_only_pts = cfg.neg_only_pts.split(",")
  else:
    neg_only_pts = None
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
    neg_only_pts = None  # do not use neg only for late riterations
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
                       neg_only_pts=neg_only_pts,
                       pos_per_asin=cfg.pos_per_asin,
                       times_negative=cfg.times_negative,
                       times_asin_negative=cfg.times_asin_negative,
                       context_per_sample=cfg.context_per_sample,
                       max_pos_pairs_per_set=cfg.max_pos_pairs_per_set,
                       output_dir=cfg.output_dir,
                       pct_dev=cfg.pct_dev)


if __name__ == "__main__":
  main()
