from contextualized_sbert.models import EntitySBERT, EntityPooling
from sentence_transformers.models import Transformer
from pathlib import Path


def load_multitask_model(model_dir: Path, task: str):
  """Load multitask model. Task can be [embedding, TODO: add more tasks]"""
  model_dir = Path(model_dir).as_posix()
  if task == "embedding":
    transformer = Transformer(model_dir)
    pooling = EntityPooling.load(Path(model_dir, "1_EntityPooling").as_posix())
    model = EntitySBERT(modules=[transformer, pooling])
  else:
    raise NotImplementedError("Model loading not implemented")

  return model
