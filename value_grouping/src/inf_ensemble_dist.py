"""Distributed self-ensemble inference based on embedding and classifier results."""

import ray
from ray import tune
import hydra

from inf_ensemble import get_product_types, run_mixed_inference_pt

# must setup ray before hydra changes working directory
ray.init(num_cpus=32)  # TODO: no easy way not to hard code this

@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.inference.ensemble

  # get all PTs that has embedding file generated
  if cfg.selected_pt:
    pts = cfg.selected_pt.strip().split(",")
  else:
    pts = get_product_types(cfg.emb_dir)

  # run distributed inference
  tune.run(run_mixed_inference_pt,
           config={
               "cfg": cfg,
               "pt": tune.grid_search(pts)
           },
           resources_per_trial={"cpu": cfg.cpu_per_job})




if __name__ == "__main__":
  main()
