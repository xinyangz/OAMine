import argparse
import os
from pathlib import Path

import ray
from ray import tune

from preprocess import get_impact_matrix


def run_chunk(config):
  data_split = config["data_split"]
  args = config["args"]
  get_impact_matrix(args, data_split, disable_tqdm=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model args
  parser.add_argument('--batch_size', default=512, type=int)
  parser.add_argument("--bert", default=None, help="Name or path to bert checkpoint")

  # Data args
  parser.add_argument('--data_dir', default=None, help="Path to catalog csv files")
  parser.add_argument('--output_dir', default=None)

  # Matrix args
  parser.add_argument('--metric',
                      default='dist',
                      help="Distance metric to use. `dist` for euclidean, `cos` for cosine.,")

  # Cuda
  parser.add_argument('--cuda', action='store_true')

  args = parser.parse_args()

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # find all input files
  pt_files = Path(args.data_dir).glob("*.csv")
  pt_files = [p.stem for p in pt_files]
  pt_files = [f"{p}.csv" for p in pt_files]
  print("Launcing perturbed masking preprocessing on:")
  print(pt_files)

  ray.init()

  tune.run(run_chunk,
           config={
               "args": args,
               "data_split": tune.grid_search(pt_files)
           },
           resources_per_trial={
               "cpu": 8,
               "gpu": 1
           })
