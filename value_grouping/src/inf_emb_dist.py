"""Distributed embedding inference w/ fine-tuned S-BERT model."""
import logging
import random
from pathlib import Path
from typing import List

import hydra
import ray
import numpy as np
from ray import tune

import utils
from contextualized_sbert import (load_multitask_model,
                                  preprocess_singleton_data)
from contextualized_sbert.data_utils import match_context

logger = logging.getLogger(__name__)

# must setup ray before hydra changes working directory
ray.init()

def run_inference_on_pts(config):
  cfg = config["cfg"]
  pts: List[str] = config["pts"]

  model_dir = Path(cfg.model_dir)
  output_dir = Path(cfg.output_dir)
  random.seed(cfg.rnd_seed)

  utils.IO.ensure_dir(output_dir)
  model = load_multitask_model(model_dir, "embedding")

  # load data from multiple PTs and combine them into
  # one large dataset for inference
  tune.report(stage="prep")

  pt2range = dict()
  pt2phrase_context_map = dict()
  all_examples = []
  for pt in pts:
    docs = utils.JsonL.load(Path(cfg.candidate_dir, f"{pt}.chunk.jsonl"))
    all_phrases = [p for doc in docs for p in doc]
    # sort by phrase popularity
    # TODO: do we add constraint for popular phrases here?
    unique_phrases = utils.Sort.unique_by_frequency(all_phrases)

    # map phrases to their contexts w/ sampling
    phrase2context_idx, contexts = match_context(unique_phrases, docs, sampling=cfg.sampling)

    # preprocess data
    examples = preprocess_singleton_data(contexts, model.tokenizer, max_seq_length=model.max_seq_length, disable_tqdm=True)

    start = len(all_examples)
    all_examples.extend(examples)
    end = len(all_examples)

    pt2range[pt] = (start, end)
    pt2phrase_context_map[pt] = phrase2context_idx

  # run BERT inference
  print(len(all_examples))
  def report_progress(progress: float):
    tune.report(progress=f"{progress*100:.2f}%", stage="inf")
  all_embeddings = model.encode(all_examples, batch_size=cfg.batch_size, show_progress_bar=False, progress_callback=report_progress)

  # gather embeddings per PT
  tune.report(stage="gather")
  for pt in pts:
    start, end = pt2range[pt]
    phrase2context_idx = pt2phrase_context_map[pt]
    pt_embeddings = all_embeddings[start:end]
    static_embeddings = dict()
    for phrase, (start, end) in phrase2context_idx.items():
      phrase_ctx_embeddings = pt_embeddings[start:end, :]
      static_embedding = phrase_ctx_embeddings.mean(0)
      static_embeddings[phrase] = static_embedding

    utils.Pkl.dump(static_embeddings, Path(output_dir, f"{pt}.emb.pkl"))

@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.inference.embedding


  # of using all PTs
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
  tune.run(run_inference_on_pts, config={
    "cfg": cfg,
    "pts": tune.grid_search(grouped_pts)
  }, resources_per_trial={"cpu": cfg.cpu_per_job, "gpu": 1})



if __name__ == "__main__":
  main()