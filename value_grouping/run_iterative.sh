TRAIN_GPU=1  # training on single gpu only
INFERENCE_GPU=0,1  # inference can be run on multiple gpu, each gpu runs inference for one PT at a time
N_INF_GPU=$(echo $INFERENCE_GPU | awk -F',' '{print NF}')  # number of comma + 1

RND_SEED=123

## SET YOUR DATASET PATH IN exp_config/dataset/your_data.yaml
## SET PRETRAINED MODEL IN exp_config/tuning/multitask.yaml
EXP_DIR=/path/to/output # where outputs will be saved
EVAL_DIR=/path/to/dev/or/test/set # ground truth evaluation clusters, either dev or test set
TRAIN_ITER=5
# CLF_INF=false  # whether to use classifier inference

USE_WANDB=false
# uncomment and set up wandb to use wandb
# export WANDB_API_KEY=2d01e86a26931a690d466139632cfdbc6c64f9eb
# export WANDB_PROJECT=oamine-release
# export WANDB_TAGS=release
# # export WANDB_MODE=offline
# USE_WANDB=true


for ((ITER=1;ITER<=$TRAIN_ITER;ITER++));
do
  PREV_ITER=$(($ITER-1))
  if (($ITER>1));
  then
    PREV_ITER_INF_DIR=$EXP_DIR/iter_$PREV_ITER/ensemble_inf
  else
    PREV_ITER_INF_DIR=null
  fi

  if [ "$ITER" == "$TRAIN_ITER" ]; then
    CLF_INF=true
  else
    CLF_INF=false
  fi

  echo =======================================================
  echo ============ Fine-tuning Data Generation ==============
  echo =======================================================

  python src/data_gen/gen_binary.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.binary.inference_dir=$PREV_ITER_INF_DIR
  python src/data_gen/gen_triplet.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.triplet.inference_dir=$PREV_ITER_INF_DIR
  python src/data_gen/gen_clf.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.clf.inference_dir=$PREV_ITER_INF_DIR
  python src/data_gen/gen_multitask.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED


  echo =======================================================
  echo =============== Multitask Fine-tuning =================
  echo =======================================================
  python src/sbert_multitask.py run.train_gpu=$TRAIN_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED


  echo =======================================================
  echo ===================== Inference =======================
  echo =======================================================
  CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/inf_emb.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
  if [ "$CLF_INF" = true ]; then
    echo "Use classifier inference"
    # by default classifier inference run on selected PTs (w/ gold labels)
    # to run full inference, set inference=full (very time consuming due to slow classifier inference)
    CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/inf_clf_dist.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
    python src/inf_ensemble_dist.py inference=selected_full run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
  else
    python src/inf_ensemble_dist.py inference=dbscan run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
  fi

  echo "Running evaluation"
  if [ "$USE_WANDB" = true ]; then
    python src/eval_clustering.py $EXP_DIR/iter_$ITER/ensemble_inf $EVAL_DIR "exp note" true
  else
    python src/eval_clustering.py $EXP_DIR/iter_$ITER/ensemble_inf $EVAL_DIR
  fi
done
