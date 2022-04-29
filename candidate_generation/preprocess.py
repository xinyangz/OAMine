"""Read a JsonL file, take the `title` field, and build the word-word impact
matrix based on pre-trained BERT.
Titles are truncated to MAX_SEQ_LENGTH."""

import argparse
import logging
import os
import pickle
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, DataCollatorWithPadding

import utils

logger = logging.getLogger(__name__)

MIN_OUTPUT_LAYER = 11
MAX_SEQ_LENGTH = 64

class MyDataset(Dataset):

  def __init__(self, tuples):
    super(MyDataset).__init__()
    self.data = tuples

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)


@dataclass
class MyCollator(DataCollatorWithPadding):

  def __call__(self, features):
    base_keys = {"input_ids", "token_type_ids"}
    base_features = [{k: v for k, v in obj.items() if k in base_keys} for obj in features]
    batch = super().__call__(base_features)
    batch["hidden_state_ids"] = torch.cat([obj["hidden_state_ids"] for obj in features])
    return batch


def find_root(parse):
  # root node"s head also == 0, so have to be removed
  for token in parse[1:]:
    if token.head == 0:
      return token.id
  return False


def _run_strip_accents(text):
  """Strips accents from a piece of text."""
  text = unicodedata.normalize("NFD", text)
  output = []
  for char in text:
    cat = unicodedata.category(char)
    if cat == "Mn":
      continue
    output.append(char)
  return "".join(output)


def match_tokenized_to_untokenized(subwords, sentence):
  token_subwords = np.zeros(len(sentence))
  sentence = [_run_strip_accents(x) for x in sentence]
  token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
  for i, subword in enumerate(subwords):
    if subword in ["[CLS]", "[SEP]", "[UNK]"]:
      continue

    while current_token_normalized is None:
      current_token_normalized = sentence[current_token].lower()

    if subword.startswith("[UNK]"):
      unk_length = int(subword[6:])
      subwords[i] = subword[:5]
      subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
    else:
      subwords_str += subword[2:] if subword.startswith("##") else subword
    if not current_token_normalized.startswith(subwords_str):
      return False

    token_ids[i] = current_token
    token_subwords[current_token] += 1
    if current_token_normalized == subwords_str:
      subwords_str = ""
      current_token += 1
      current_token_normalized = None

  assert current_token_normalized is None
  while current_token < len(sentence):
    assert not sentence[current_token]
    current_token += 1
  assert current_token == len(sentence)

  return token_ids


def get_all_subword_id(mapping, idx):
  current_id = mapping[idx]
  id_for_all_subwords = [tmp_id for tmp_id, v in enumerate(mapping) if v == current_id]
  return id_for_all_subwords


def merge_subword_tokens(tokens: List[str]):
  new_tokens = []
  last_token = ""
  for tok in tokens:
    if tok.startswith("##"):
      last_token += tok[2:]
      continue
    else:
      if last_token:
        new_tokens.append(last_token)
      last_token = tok
  if last_token:
    new_tokens.append(last_token)
  return new_tokens


def load_data(data_dir, data_split, tokenizer, disable_tqdm=False):
  logger.info("Loading data")
  docs = utils.JsonL.load(Path(data_dir, f"{data_split}.jsonl"))
  tokenized_texts = []
  asins = []

  for doc in tqdm(docs, desc="Load & tokenize", disable=disable_tqdm):
    if doc["title"] is None or len(doc["title"]) < 3:
      continue
    subword_tokens = tokenizer.tokenize(doc["title"])
    tokenized_texts.append(merge_subword_tokens(subword_tokens))
    asins.append(doc["asin"])
  return asins, tokenized_texts


def get_impact_matrix(args, data_split, disable_tqdm=False):
  tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=True)
  asins, texts = load_data(args.data_dir, data_split, tokenizer, disable_tqdm=disable_tqdm)

  mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

  # data preprocessing
  logger.info("Preprocessing for inference")
  dataset = []  # raw batches, len = #sent * #token_in_each_sent
  tokenized_texts = []  # len = #sent
  sent2batch = []  # [[0,1,2], [3,4,5,6,7], ...]
  processed_asins = []
  processed_texts = []
  nu = 0
  for asin, sents in zip(asins, tqdm(texts, "Preprocess", disable=disable_tqdm)):
    sentence = " ".join(sents)
    tokenized_text = tokenizer.tokenize(sentence)
    tokenized_text.insert(0, "[CLS]")
    tokenized_text.append("[SEP]")
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    try:
      mapping = match_tokenized_to_untokenized(tokenized_text, sents)
    except Exception as e:
      continue
    if not mapping:
      continue
    processed_asins.append(asin)
    processed_texts.append(sents)

    # 1. Generate mask indices
    logger.info("Generate mask indices")
    sent_ids = []
    for i in range(0, len(tokenized_text)):
      id_for_all_i_tokens = get_all_subword_id(mapping, i)
      tmp_indexed_tokens = list(indexed_tokens)
      for tmp_id in id_for_all_i_tokens:
        if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
          tmp_indexed_tokens[tmp_id] = mask_id
      one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
      for j in range(0, len(tokenized_text)):
        id_for_all_j_tokens = get_all_subword_id(mapping, j)
        for tmp_id in id_for_all_j_tokens:
          if mapping[tmp_id] != -1:
            one_batch[j][tmp_id] = mask_id

      # 2. Convert one batch to PyTorch tensors
      logger.info("Convert to torch tensors")
      dataset.extend([{
          "input_ids": torch.tensor(input_ids),
          "token_type_ids": torch.tensor([0 for _ in input_ids]),
          "hidden_state_ids": torch.tensor([i])
      } for input_ids in one_batch])
      sent_ids.append((nu, nu + len(one_batch)))
      nu += len(one_batch)
    sent2batch.append(sent_ids)
    tokenized_texts.append(tokenized_text)

  # inference
  logger.info("Run inference")
  model = BertModel.from_pretrained(args.bert, output_hidden_states=True)
  model.eval()

  # if this doesn"t work for your model, adapt it accordingly
  LAYER = len(model.encoder.layer)
  LAYER += 1  # also consider embedding layer
  out = [[] for _ in range(LAYER)]

  n_device = 1
  if args.cuda:
    n_device = torch.cuda.device_count()
    if n_device > 1:
      # Comments on DataParallel
      # this implementation doesn"t seem very efficient on multiple GPU
      # prefer launching separate jobs for separate files
      raise NotImplementedError("Data parallel turned off")
    model.half()
    model.to("cuda")


  assert len(sent2batch) == len(processed_asins), f"{len(sent2batch)}, {len(processed_asins)}"
  dataset = MyDataset(dataset)
  data_loader = DataLoader(dataset, batch_size=args.batch_size*n_device, num_workers=8, collate_fn=MyCollator(tokenizer=tokenizer, padding=True, max_length=MAX_SEQ_LENGTH, pad_to_multiple_of=8))

  inference_results = [None] * LAYER
  for batch in tqdm(data_loader, desc="Inference", disable=disable_tqdm):
    with torch.no_grad():
      if args.cuda:
        batch = batch.to("cuda")
      model_outputs = model(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"])
      all_layers = model_outputs[-1]  # 12 layers + embedding layer
      for i, layer_output in enumerate(all_layers):
        if i < MIN_OUTPUT_LAYER:
          continue
        layer_output = list(layer_output[range(len(layer_output)), batch["hidden_state_ids"], :].cpu().numpy())
        if inference_results[i]:
          inference_results[i].extend(layer_output)
        else:
          inference_results[i] = layer_output

  # free up gpu memory
  del model
  del data_loader
  del dataset
  torch.cuda.empty_cache()

  # compute impact matrix
  logger.info("Generating impact matrices")
  for k, layer_inference_results in enumerate(tqdm(inference_results, desc="Layer", disable=disable_tqdm)):
    if k < MIN_OUTPUT_LAYER:
      continue

    # for sentence
    for i_sent, (sent_indices, tokenized_text) in enumerate(zip(tqdm(sent2batch, desc="Sentence", disable=disable_tqdm), tokenized_texts)):
      assert len(sent_indices) == len(tokenized_text), f"{len(sent_indices)} {len(tokenized_text)}"
      sent_len = len(tokenized_text)
      init_matrix = np.zeros((sent_len, sent_len))

      # for token in sentence
      for i, inference_start_end in enumerate(sent_indices):
        start, end = inference_start_end
        hidden_states = layer_inference_results[start:end]
        base_state = hidden_states[i]
        for j, state in enumerate(hidden_states):
          if args.metric == "dist":
            init_matrix[i][j] = np.linalg.norm(base_state - state)
          if args.metric == "cos":
            init_matrix[i][j] = np.dot(base_state, state) / (np.linalg.norm(base_state) * np.linalg.norm(state))
      out[k].append((processed_asins[i_sent], processed_texts[i_sent], tokenized_text, init_matrix))

  for k, one_layer_out in enumerate(out):
    if k < MIN_OUTPUT_LAYER:
      continue
    pt_name = data_split
    k_output = Path(args.output_dir, f"{args.metric}-{pt_name}-{k}.pkl")
    with open(k_output, "wb") as fout:
      pickle.dump(out[k], fout)
      fout.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Model args
  parser.add_argument("--batch_size", default=512, type=int)
  parser.add_argument("--bert", default=None, help="Name or path to BERT checkpoint")

  # Data args
  parser.add_argument("--data_split", default=None)
  parser.add_argument("--data_dir", default=None)
  parser.add_argument("--output_dir", default=None)

  # Matrix args
  parser.add_argument("--metric", default="dist")

  # Cuda
  parser.add_argument("--cuda", action="store_true")

  # Debug
  parser.add_argument("--no_tqdm", action="store_true")

  args = parser.parse_args()

  utils.IO.ensure_dir(args.output_dir)

  get_impact_matrix(args, args.data_split, disable_tqdm=args.no_tqdm)
