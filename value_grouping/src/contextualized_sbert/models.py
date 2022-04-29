import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import utils
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.models import Pooling
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (paired_cosine_distances,
                                      paired_euclidean_distances,
                                      paired_manhattan_distances)
from tqdm.auto import trange
from transformers.tokenization_utils_base import BatchEncoding

from contextualized_sbert.data import EncodedEntityContext, EncodedTuple

logger = logging.getLogger(__name__)


class ContextualizedBinaryClfEvaluator(BinaryClassificationEvaluator):

  def compute_metrices(self, model):
    sentences = self.sentences1 + self.sentences2
    embeddings = model.encode(sentences,
                              batch_size=self.batch_size,
                              show_progress_bar=self.show_progress_bar,
                              convert_to_numpy=True)
    n_half = len(self.sentences1)
    embeddings1 = embeddings[:n_half]
    embeddings2 = embeddings[n_half:]

    cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

    embeddings1_np = np.asarray(embeddings1)
    embeddings2_np = np.asarray(embeddings2)
    dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

    labels = np.asarray(self.labels)
    output_scores = {}
    for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True],
                                              ['manhatten', 'Manhatten-Distance', manhattan_distances, False],
                                              ['euclidean', 'Euclidean-Distance', euclidean_distances, False],
                                              ['dot', 'Dot-Product', dot_scores, True]]:
      acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
      f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
      ap = average_precision_score(labels, scores * (1 if reverse else -1))

      logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
      logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
      logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
      logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
      logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

      output_scores[short_name] = {
          'accuracy': acc,
          'accuracy_threshold': acc_threshold,
          'f1': f1,
          'f1_threshold': f1_threshold,
          'precision': precision,
          'recall': recall,
          'ap': ap
      }

    return output_scores

  @classmethod
  def from_input_examples(cls, examples: List[EncodedTuple], **kwargs):
    sentences1 = []
    sentences2 = []
    scores = []

    for example in examples:
      sentences1.append(example[0])
      sentences2.append(example[1])
      scores.append(int(example[2]))
    return cls(sentences1, sentences2, scores, **kwargs)


class Classifier(nn.Module):
  def __init__(self, emb_dim: int, num_labels: int, clf_dropout: float = 0.):
    super(Classifier, self).__init__()
    self.emb_dim = emb_dim
    self.num_labels = num_labels
    self.clf_dropout = clf_dropout
    self.config_keys = ["emb_dim", "num_labels", "clf_dropout"]

    self.dropout = nn.Dropout(clf_dropout)
    self.classifier = nn.Linear(emb_dim, num_labels)
    # init clf layer
    self.classifier.weight.data.normal_(mean=0.0, std=1.)
    self.classifier.bias.data.zero_()

  def forward(self, features: Dict[str, torch.Tensor]):
    embs = features['sentence_embedding']
    clf_output = self.dropout(embs)
    clf_output = self.classifier(clf_output)
    features["classifier_logits"] = clf_output
    return features

  def get_config_dict(self):
    return {key: self.__dict__[key] for key in self.config_keys}

  def save(self, output_path):
    utils.IO.ensure_dir(output_path)
    with open(Path(output_path, 'config.json'), 'w') as fOut:
      json.dump(self.get_config_dict(), fOut, indent=2)
    torch.save(self.state_dict(), Path(output_path, "model.pt"))

  @classmethod
  def load(cls, input_path):
    with open(Path(input_path, 'config.json')) as fIn:
      config = json.load(fIn)
    obj = cls(**config)
    obj.load_state_dict(torch.load(Path(input_path, "model.pt"), map_location="cpu"))
    return obj


class ClfLoss(nn.Module):

  def __init__(self, model: SentenceTransformer):
    super(ClfLoss, self).__init__()
    self.model = model
    self.loss_fct = nn.CrossEntropyLoss()

  def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
    reps = [self.model(sentence_feature)['classifier_logits'] for sentence_feature in sentence_features]
    output = reps[0]

    if labels is not None:
      loss = self.loss_fct(output, labels.view(-1))
      return loss
    else:
      return output


class ClassifierClfLoss(nn.Module):

  def __init__(self, model: SentenceTransformer, emb_dim, num_labels, clf_dropout):
    super(ClassifierClfLoss, self).__init__()
    self.model = model
    self.loss_fct = nn.CrossEntropyLoss()
    self.classifier = Classifier(emb_dim, num_labels, clf_dropout)

  def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
    reps = [self.classifier(self.model(sentence_feature))["classifier_logits"] for sentence_feature in sentence_features]
    # reps = [self.model(sentence_feature)['classifier_logits'] for sentence_feature in sentence_features]
    output = reps[0]

    if labels is not None:
      loss = self.loss_fct(output, labels.view(-1))
      return loss
    else:
      return output

  def save(self, path):
    self.classifier.save(path)


class EntitySBERT(SentenceTransformer):

  def __init__(self, model_name_or_path=None, modules=None, device=None, cache_folder=None):
    super().__init__(model_name_or_path=model_name_or_path, modules=modules, device=device, cache_folder=cache_folder)
    self.additional_components = []

  def _pad_to_tensor(self, features) -> BatchEncoding:
    # remove pooling masks
    pooling_masks = []
    hf_features = []
    for feature in features:
      pooling_masks.append(feature["pooling_mask"])
      hf_features.append({k: v for k, v in feature.items() if k != "pooling_mask"})
    batch_tensor = self.tokenizer.pad(
        hf_features,
        padding=True,
        max_length=64,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    # manually pad pooling masks
    padded_length = len(batch_tensor["input_ids"][0])
    pooling_masks = [mask + [0] * (padded_length - len(mask)) for mask in pooling_masks]
    try:
      for pooling_mask, input_id in zip(pooling_masks, batch_tensor["input_ids"]):
        assert len(pooling_mask) == len(input_id), f"{pooling_mask}, {input_id}"
      pooling_masks = torch.tensor(pooling_masks).float()
    except Exception as e:
      print(e)
      for pooling_mask in pooling_masks:
        print(len(pooling_mask))
      print()
      for input_ids in batch_tensor["input_ids"]:
        print(len(input_ids))
    batch_tensor["pooling_mask"] = pooling_masks
    return batch_tensor

  def smart_batching_collate(self, batch: List[EncodedTuple]):
    # sentence_features: a list of dictionary
    # batch is either triplet or tuple with labels
    is_triplet = False
    is_clf = False
    if isinstance(batch[0], dict):  # clf inference
      is_clf = True
      batch_tensor = self._pad_to_tensor(batch).to(self._target_device)
      labels = None
      sentence_features = [batch_tensor]
    elif len(batch[0]) == 2:  # clf
      is_clf = True
      sent_features, labels = list(zip(*batch))
      batch_tensor_0 = self._pad_to_tensor(sent_features).to(self._target_device)
      labels = torch.tensor(labels).to(self._target_device)
      sentence_features = [batch_tensor_0]
    elif len(batch[0]) == 3:  # binary or triplet
      if isinstance(batch[0][2], int) or isinstance(batch[0][2], float):
        sent_0_features, sent_1_features, labels = list(zip(*batch))
      else:
        is_triplet = True
        sent_0_features, sent_1_features, sent_2_features = list(zip(*batch))

      batch_tensor_0 = self._pad_to_tensor(sent_0_features).to(self._target_device)
      batch_tensor_1 = self._pad_to_tensor(sent_1_features).to(self._target_device)

      if is_triplet:
        batch_tensor_2 = self._pad_to_tensor(sent_2_features).to(self._target_device)
        sentence_features = [batch_tensor_0, batch_tensor_1, batch_tensor_2]
        labels = None
      else:
        labels = torch.tensor(labels).to(self._target_device)
        sentence_features = [
            batch_tensor_0,
            batch_tensor_1,
        ]
    return sentence_features, labels

  def add_save_component(self, component, name):
    self.additional_components.append((name, component))

  def save(self, path: str, model_name=None, create_model_card=True):
    super().save(path, model_name, create_model_card)
    for name, component in self.additional_components:
      component.save(Path(path, name))


  def encode(self,
             examples: List[EncodedEntityContext],
             batch_size: int = 32,
             show_progress_bar: bool = None,
             progress_callback=None,
             output_value: str = 'sentence_embedding',
             convert_to_numpy: bool = True,
             convert_to_tensor: bool = False,
             device: str = None,
             normalize_embeddings: bool = False) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
    self.eval()
    if show_progress_bar is None:
      show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

    if convert_to_tensor:
      convert_to_numpy = False

    if device is None:
      device = self._target_device

    self.to(device)

    all_embeddings = []
    # oddly enough, the original _text_length works even for our new data format
    length_sorted_idx = np.argsort([-self._text_length(sen) for sen in examples])
    examples_sorted = [examples[idx] for idx in length_sorted_idx]

    last_progress = 0
    for start_index in trange(0, len(examples), batch_size, desc="Batches", disable=not show_progress_bar):
      examples_batch = examples_sorted[start_index:start_index + batch_size]
      features = self._pad_to_tensor(examples_batch).to(device)

      progress = start_index / len(examples)
      if progress - last_progress > 0.05:
        last_progress = progress
        if progress_callback is not None:
          progress_callback(progress)

      with torch.no_grad():
        out_features = self.forward(features)

        if output_value == 'token_embeddings':
          embeddings = []
          for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
            last_mask_id = len(attention) - 1
            while last_mask_id > 0 and attention[last_mask_id].item() == 0:
              last_mask_id -= 1

            embeddings.append(token_emb[0:last_mask_id + 1])
        else:  #Sentence embeddings
          embeddings = out_features[output_value]
          embeddings = embeddings.detach()
          if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

          # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
          if convert_to_numpy:
            embeddings = embeddings.cpu()

        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_tensor:
      all_embeddings = torch.stack(all_embeddings)
    elif convert_to_numpy:
      all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

    return all_embeddings


class EntityPooling(nn.Module):

  def __init__(self, word_embedding_dimension: int):
    super(EntityPooling, self).__init__()

    self.word_embedding_dimension = word_embedding_dimension
    self.pooling_output_dimension = word_embedding_dimension
    self.config_keys = ["word_embedding_dimension"]

  def __repr__(self):
    return "EntityPooling({})".format(self.get_config_dict())

  def forward(self, features: Dict[str, torch.Tensor]):
    token_embeddings = features['token_embeddings']
    cls_token = features['cls_token_embeddings']
    attention_mask = features['attention_mask']
    pooling_mask = features['pooling_mask']

    masked = token_embeddings * pooling_mask.unsqueeze(-1)
    sum_embeddings = torch.sum(masked, 1)
    sum_mask = pooling_mask.sum(1, keepdim=True)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    output_vector = sum_embeddings / sum_mask

    features.update({'sentence_embedding': output_vector})
    return features

  def get_pooling_mode_str(self) -> str:
    return "entity"

  def get_config_dict(self):
    return {key: self.__dict__[key] for key in self.config_keys}

  def get_sentence_embedding_dimension(self):
    return self.pooling_output_dimension

  def save(self, output_path):
    with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
      json.dump(self.get_config_dict(), fOut, indent=2)

  @classmethod
  def load(cls, input_path):
    with open(os.path.join(input_path, 'config.json')) as fIn:
      config = json.load(fIn)

    return cls(**config)
