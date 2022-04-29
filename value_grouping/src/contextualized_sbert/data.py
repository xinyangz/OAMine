from typing import List, Tuple, Dict, Any
from collections import Counter, namedtuple
from transformers.tokenization_utils import PreTrainedTokenizer
import logging
from tqdm.auto import tqdm
import utils
import random
from contextualized_sbert.data_utils import EntityContext

logger = logging.getLogger(__name__)


ContextualizedExample = namedtuple("ContextualizedExample", ["entities", "label"])
EncodedEntityContext = Dict[str, List[int]]
EncodedTuple = Tuple[EncodedEntityContext, EncodedEntityContext, float]
MAX_SEQ_LENGTH = 64


def parse_tokenizer_output(tokenizer_output, max_seq_length, disable_tqdm=False):
  """Take tokenized output from sequence of left_ctx, entity, right_ctx and generate
  encoded examples with pooling masks"""
  all_input_ids = tokenizer_output.input_ids
  dataset = []
  assert len(all_input_ids) % 3 == 0
  n_combined = len(all_input_ids) // 3

  for i in tqdm(range(n_combined), desc="Preprocess", disable=disable_tqdm):
    left = all_input_ids[i * 3]
    entity = all_input_ids[i * 3 + 1]
    right = all_input_ids[i * 3 + 2]

    # truncation
    token_type_ids = []
    attention_mask = []
    l_start = 1
    l_end = len(left) - 1
    r_start = 1
    r_end = len(right) - 1
    trunc_entity = False
    total_length = l_end - l_start + r_end - r_start + len(entity)
    if total_length > max_seq_length:
      quota_minus_entity = max_seq_length - len(entity)
      if quota_minus_entity < 0:
        # logger.warning(f"Entity {i} too long")
        l_start = l_end
        r_end = r_start
        trunc_entity = True
      else:
        quota_left = quota_right = quota_minus_entity // 2
        if quota_left + quota_right != quota_minus_entity:
          quota_left += 1
        l_start = max(0, l_end - quota_left)
        r_end = min(r_start + quota_right, len(right))

    if trunc_entity:
      end_token = entity[-1]
      entity = entity[:max_seq_length - 1]
      entity.append(end_token)

    input_ids = [entity[0]] + left[l_start:l_end] + entity[1:-1] + right[r_start:r_end] + [entity[-1]]
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    # keep only entity repr minus special tokens
    pooling_mask = [0] * (l_end - l_start + 1) + [1] * (len(entity) - 2) + [0] * (r_end - r_start + 1)

    assert len(pooling_mask) == len(input_ids), f"{len(input_ids)}, {len(entity)}, {len(pooling_mask)}"

    dataset.append({
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "pooling_mask": pooling_mask,
    })
  return dataset


def parse_clf_tokenizer_output(entity_tokenizer_output, pt_tokenizer_output, max_seq_length, disable_tqdm=False):
  """Take tokenized output from sequence of left_ctx, entity, right_ctx, pt_name and generate
  encoded examples with pooling masks"""
  entity_input_ids = entity_tokenizer_output.input_ids
  pt_input_ids = pt_tokenizer_output.input_ids
  dataset = []
  assert len(entity_input_ids) % 3 == 0
  n_combined = len(entity_input_ids) // 3

  for i in tqdm(range(n_combined), desc="Preprocess", disable=disable_tqdm):
    # truncation strategy: keep entity, pt, truncate left and right context
    left = entity_input_ids[i * 3]
    entity = entity_input_ids[i * 3 + 1]
    right = entity_input_ids[i * 3 + 2]
    pt_name = pt_input_ids[i]

    token_type_ids = []
    # special token only count once
    l_start = 1
    l_end = len(left) - 1
    r_start = 1
    r_end = len(right) - 1
    pt_start = 1
    pt_end = len(pt_name) - 1
    trunc_entity = False

    # entity has [CLS] and [SEP], add one [SEP] for pt, entity seperation
    total_length = l_end - l_start + r_end - r_start + pt_end - pt_start + len(entity) + 1
    if total_length > max_seq_length:
      quota_minus_entity = max_seq_length - len(entity) - 1
      if quota_minus_entity < 0:
        # worst case, entity too long, keep only entity
        l_start = l_end
        r_end = r_start
        pt_end = pt_start
        trunc_entity = True
      else:
        # next, see if we have enough room for pt_name
        if len(pt_name) - 2 > quota_minus_entity:
          # pt_name too long, truncate, throw away left and right context
          pt_end = min(pt_start + quota_minus_entity, pt_end)
          l_start = l_end
          r_end = r_start
        else:
          # enough room for pt_name and both contexts, figure out
          # context truncation
          quota_context = quota_minus_entity - len(pt_name) - 2
          quota_left = quota_right = quota_context // 2
          if quota_left + quota_right != quota_minus_entity:
            quota_left += 1
          l_start = max(l_start, l_end - quota_left)
          r_end = min(r_start + quota_right, r_end)

    if trunc_entity:
      end_token = entity[-1]
      entity = entity[:max_seq_length - 2]
      entity.append(end_token)

    # [CLS] left_ctx entity right_ctx [SEP] pt_name [SEP]
    CLS = entity[0]
    SEP = entity[-1]
    input_ids = [CLS] + left[l_start:l_end] + entity[1:-1] + right[r_start:r_end] + [SEP] + pt_name[pt_start:pt_end] + [SEP]
    type_0_len = 1 + l_end - l_start + len(entity) - 2 + r_end - r_start + 1
    type_1_len = len(input_ids) - type_0_len
    token_type_ids = [0] * type_0_len + [1] * type_1_len
    attention_mask = [1] * len(input_ids)
    # keep only entity repr minus special tokens
    pooling_mask = [0] * (l_end - l_start + 1) + [1] * (len(entity) - 2) + [0] * (r_end - r_start + 1) + [0] * (pt_end - pt_start + 1)

    assert len(pooling_mask) == len(input_ids), f"{len(input_ids)}, {len(entity)}, {len(pooling_mask)}"
    assert len(input_ids) <= max_seq_length

    dataset.append({
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "pooling_mask": pooling_mask,
    })
  return dataset


# tokenize and mask
def preprocess_pairwise_data(samples: List[ContextualizedExample], tokenizer: PreTrainedTokenizer, max_seq_length=64, disable_tqdm=False):
  raw_sentences = []
  for sample in samples:
    ent_ctx_a, ent_ctx_b = sample.entities
    raw_sentences.extend([ent_ctx_a.left_context, ent_ctx_a.entity, ent_ctx_a.right_context])
    raw_sentences.extend([ent_ctx_b.left_context, ent_ctx_b.entity, ent_ctx_b.right_context])

  tokenizer_output = tokenizer(raw_sentences, truncation="do_not_truncate")

  dataset = parse_tokenizer_output(tokenizer_output, max_seq_length, disable_tqdm=disable_tqdm)

  dataset = [(dataset[2 * i], dataset[2 * i + 1], samples[i].label) for i in range(len(samples))]
  # DEBUG
  # for d in dataset:
  #   print(tokenizer.convert_ids_to_tokens(d[0]["input_ids"]))
  #   print(tokenizer.convert_ids_to_tokens(d[1]["input_ids"]))
  #   print()
  # exit()
  return dataset


def preprocess_triplet_data(samples: List[Tuple[EntityContext, EntityContext, EntityContext]],
                            tokenizer: PreTrainedTokenizer,
                            max_seq_length=64,
                            disable_tqdm=False):
  raw_sentences = []
  for sample in samples:
    ent_ctx_a, ent_ctx_b, ent_ctx_c = sample
    raw_sentences.extend([ent_ctx_a.left_context, ent_ctx_a.entity, ent_ctx_a.right_context])
    raw_sentences.extend([ent_ctx_b.left_context, ent_ctx_b.entity, ent_ctx_b.right_context])
    raw_sentences.extend([ent_ctx_c.left_context, ent_ctx_c.entity, ent_ctx_c.right_context])

  tokenizer_output = tokenizer(raw_sentences, truncation="do_not_truncate")

  dataset = parse_tokenizer_output(tokenizer_output, max_seq_length, disable_tqdm=disable_tqdm)

  dataset = [(dataset[3 * i], dataset[3 * i + 1], dataset[3 * i + 2]) for i in range(len(samples))]
  # DEBUG
  # for d in dataset:
  #   print(tokenizer.convert_ids_to_tokens(d[0]["input_ids"]))
  #   print(tokenizer.convert_ids_to_tokens(d[1]["input_ids"]))
  #   print()
  # exit()
  return dataset


def preprocess_clf_data(samples: List[Tuple[EntityContext, str, str]], tokenizer: PreTrainedTokenizer, label_map=None, max_seq_length=64, disable_tqdm=False):
  raw_sentences = []
  pt_names = []
  labels = []
  has_labels = True
  sample_0 = samples[0]
  if len(sample_0) == 2:
      has_labels = False
  for sample in samples:
    if has_labels:
      entity, pt_name, label = sample
    else:
      entity, pt_name = sample
    raw_sentences.extend([entity.left_context, entity.entity, entity.right_context])
    pt_names.append(pt_name)
    if has_labels:
      labels.append(label)

  if has_labels and label_map is None:
    label_map = dict()
    label_set = set(labels)
    for i, label in enumerate(label_set):
      label_map[label] = i

  entity_tokenizer_output = tokenizer(raw_sentences, truncation="do_not_truncate")
  pt_tokenizer_output = tokenizer(pt_names, truncation="do_not_truncate")

  dataset = parse_clf_tokenizer_output(entity_tokenizer_output, pt_tokenizer_output, max_seq_length, disable_tqdm=disable_tqdm)
  if has_labels:
    assert len(dataset) == len(labels)
    dataset = [(example, label_map[labels[i]]) for i, example in enumerate(dataset)]

  # DEBUG
  # for d in dataset:
  #   print(tokenizer.convert_ids_to_tokens(d[0]["input_ids"]))
  #   print(d[0]["token_type_ids"])
  #   print(d[0]["pooling_mask"])
  #   exit()

  return dataset, label_map


def preprocess_singleton_data(samples: List[EntityContext], tokenizer: PreTrainedTokenizer, max_seq_length=64, disable_tqdm=False):
  raw_sentences = []
  for ent_ctx in samples:
    raw_sentences.extend([ent_ctx.left_context, ent_ctx.entity, ent_ctx.right_context])
  tokenizer_output = tokenizer(raw_sentences, truncation="do_not_truncate")
  dataset = parse_tokenizer_output(tokenizer_output, max_seq_length, disable_tqdm=disable_tqdm)
  return dataset


def load_triplet_dataset(train_file,
                         tokenizer,
                         dev_file=None,
                         train_limit=None,
                         dev_limit=None,
                         max_seq_length=MAX_SEQ_LENGTH):
  logger.info("Read triplet dataset")

  train_samples = []
  dev_samples = []

  train_objs = utils.JsonL.load(train_file)
  # DEBUG
  if train_limit:
    random.shuffle(train_objs)
    train_objs = train_objs[:train_limit]
    # train_objs = train_objs[:1000]
  for obj in train_objs:
    ent_a = obj["anchor"]
    ent_b = obj["positive"]
    ent_c = obj["negative"]
    ent_a = EntityContext(left_context=ent_a["left_context"],
                          entity=ent_a["entity"],
                          right_context=ent_a["right_context"])
    ent_b = EntityContext(left_context=ent_b["left_context"],
                          entity=ent_b["entity"],
                          right_context=ent_b["right_context"])
    ent_c = EntityContext(left_context=ent_c["left_context"],
                          entity=ent_c["entity"],
                          right_context=ent_c["right_context"])
    train_samples.append((ent_a, ent_b, ent_c))
  train_dataset = preprocess_triplet_data(train_samples, tokenizer, max_seq_length=max_seq_length)

  if dev_file:
    dev_objs = utils.JsonL.load(dev_file)
    # DEBUG
    if dev_limit:
      random.shuffle(dev_objs)
      dev_objs = dev_objs[:dev_limit]
    for obj in dev_objs:
      ent_a = obj["entity_a"]
      ent_b = obj["entity_b"]
      ent_a = EntityContext(left_context=ent_a["left_context"],
                            entity=ent_a["entity"],
                            right_context=ent_a["right_context"])
      ent_b = EntityContext(left_context=ent_b["left_context"],
                            entity=ent_b["entity"],
                            right_context=ent_b["right_context"])
      raw_label = int(obj["label"])
      if raw_label > 0:
        label = 1
      else:
        label = 0
      dev_samples.append(ContextualizedExample(entities=[ent_a, ent_b], label=label))
    dev_dataset = preprocess_pairwise_data(dev_samples, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    return train_dataset, dev_dataset

  return train_dataset


def load_binary_dataset(train_file,
                        tokenizer,
                        dev_file=None,
                        train_limit=None,
                        dev_limit=None,
                        max_seq_length=MAX_SEQ_LENGTH):
  # Convert the dataset to a DataLoader ready for training
  logger.info("Read binary dataset")

  train_samples = []
  dev_samples = []

  train_objs = utils.JsonL.load(train_file)

  # DEBUG
  if train_limit:
    random.shuffle(train_objs)
    train_objs = train_objs[:train_limit]
  for obj in train_objs:
    ent_a = obj["entity_a"]
    ent_b = obj["entity_b"]
    ent_a = EntityContext(left_context=ent_a["left_context"],
                          entity=ent_a["entity"],
                          right_context=ent_a["right_context"])
    ent_b = EntityContext(left_context=ent_b["left_context"],
                          entity=ent_b["entity"],
                          right_context=ent_b["right_context"])
    raw_label = int(obj["label"])
    if raw_label > 0:
      label = 1.
    else:
      label = 0.
    train_samples.append(ContextualizedExample(entities=[ent_a, ent_b], label=label))
  train_dataset = preprocess_pairwise_data(train_samples, tokenizer, max_seq_length=MAX_SEQ_LENGTH)

  if dev_file:
    dev_objs = utils.JsonL.load(dev_file)
    # DEBUG
    if dev_limit:
      random.shuffle(dev_objs)
      dev_objs = dev_objs[:dev_limit]
    for obj in dev_objs:
      ent_a = obj["entity_a"]
      ent_b = obj["entity_b"]
      ent_a = EntityContext(left_context=ent_a["left_context"],
                            entity=ent_a["entity"],
                            right_context=ent_a["right_context"])
      ent_b = EntityContext(left_context=ent_b["left_context"],
                            entity=ent_b["entity"],
                            right_context=ent_b["right_context"])
      raw_label = int(obj["label"])
      if raw_label > 0:
        label = 1
      else:
        label = 0
      dev_samples.append(ContextualizedExample(entities=[ent_a, ent_b], label=label))
    dev_dataset = preprocess_pairwise_data(dev_samples, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    return train_dataset, dev_dataset

  return train_dataset


def load_clf_dataset(train_file,
                     tokenizer,
                     train_limit=None,
                     dev_limit=None,
                     dev_file=None,
                     max_seq_length=MAX_SEQ_LENGTH):
  logger.info("Read classification dataset")

  train_samples = []
  dev_samples = []

  train_objs = utils.JsonL.load(train_file)
  # DEBUG
  if train_limit:
    random.shuffle(train_objs)
    train_objs = train_objs[:train_limit]
    # train_objs = train_objs[:1000]
  for obj in train_objs:
    entity_context = EntityContext(left_context=obj["left_context"], entity=obj["entity"], right_context=obj["right_context"])
    pt_name = obj["pt_name"]
    label = obj["label"]
    train_samples.append((entity_context, pt_name, label))
  train_dataset, label_map = preprocess_clf_data(train_samples, tokenizer, max_seq_length=max_seq_length)

  if dev_file:
    dev_objs = utils.JsonL.load(dev_file)
    # DEBUG
    if dev_limit:
      random.shuffle(dev_objs)
      dev_objs = dev_objs[:dev_limit]
    for obj in dev_objs:
      entity_context = EntityContext(left_context=obj["left_context"], entity=obj["entity"], right_context=obj["right_context"])
      pt_name = obj["pt_name"]
      label = obj["label"]
      dev_samples.append((entity_context, pt_name, label))
    dev_dataset = preprocess_pairwise_data(dev_samples, tokenizer, label_map=label_map, max_seq_length=MAX_SEQ_LENGTH)
    return train_dataset, dev_dataset, label_map

  return train_dataset, label_map
