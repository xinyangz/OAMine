from pathlib import Path
import json
from typing import List, Union, Any
import io


def load_input(INPUT_FILE: Path):
  docs = []
  with Path(INPUT_FILE).open("r") as f:
    for line in f:
      docs.append(json.loads(line))
  return docs

def save_output(OUTPUT_FILE: Path, docs: List[str]):
  with OUTPUT_FILE.open("w") as f:
    for doc in docs:
      f.write(json.dumps(doc) + "\n")


class IO:

  @classmethod
  def ensure_dir(cls, dir: Union[str, Path], parents=True, exist_ok=False):
    if not Path(dir).is_dir():
      Path(dir).mkdir(parents=parents, exist_ok=exist_ok)


class JsonL(IO):

  @classmethod
  def load(cls, input_file: Path):
    docs = []
    with Path(input_file).open("r") as f:
      for line in f:
        docs.append(json.loads(line))
    return docs

  @classmethod
  def save(cls, output_file: Union[str, Path], docs: List[Any]):
    output_file = Path(output_file)
    cls.ensure_dir(output_file.parent)
    with output_file.open("w") as f:
      for doc in docs:
        f.write(json.dumps(doc) + "\n")


class TextIO(IO):

  @classmethod
  def save(cls, output_file: Union[str, Path, io.IOBase], docs: List[str]):
    if isinstance(output_file, io.IOBase):
      f = output_file
      for doc in docs:
        f.write(doc + "\n")
    else:
      output_file = Path(output_file)
      cls.ensure_dir(output_file.parent)
      with output_file.open("w") as f:
        for doc in docs:
          f.write(doc + "\n")

  @classmethod
  def load_lines(cls, fname, use_int=False):
    docs = []
    with open(fname, "r") as f:
      for line in f:
        if use_int:
          docs.append(int(line.strip()))
        else:
          docs.append(line.strip())
    return docs
