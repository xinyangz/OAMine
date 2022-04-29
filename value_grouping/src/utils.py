import io
import json
import logging
import math
import pickle as pkl
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, List, Union


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


class CustomFormatter(logging.Formatter):
  """Logging Formatter to add colors and count warning / errors"""

  grey = "\x1b[38;21m"
  yellow = "\x1b[33;21m"
  red = "\x1b[31;21m"
  bold_red = "\x1b[31;1m"
  reset = "\x1b[0m"
  log_format = "%(asctime)s [%(levelname)s] %(module)s - %(funcName)s: %(message)s"
  datefmt = '%Y-%m-%d %H:%M:%S'

  FORMATS = {
      logging.DEBUG: grey + log_format + reset,
      logging.INFO: grey + log_format + reset,
      logging.WARNING: yellow + log_format + reset,
      logging.ERROR: red + log_format + reset,
      logging.CRITICAL: bold_red + log_format + reset
  }

  def __init__(self, fmt, datefmt) -> None:
    super().__init__(fmt=fmt, datefmt=datefmt)
    self.log_format = fmt
    self.datefmt = datefmt

  def format(self, record):
    log_fmt = self.FORMATS.get(record.levelno)
    formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
    return formatter.format(record)


class Log:

  @classmethod
  def get_logger(cls, name):
    return logging.getLogger(name)

  @classmethod
  def config_logging(cls,
                     level=logging.INFO,
                     log_format="%(asctime)s [%(levelname)s] %(module)s - %(funcName)s: %(message)s",
                     datefmt='%Y-%m-%d %H:%M:%S'):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter(fmt=log_format, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[ch])


class Pkl:

  @classmethod
  def dump(cls, obj, fname: Union[str, Path]):
    p = Path(fname)
    IO.ensure_dir(p.parent)
    with p.open("wb") as f:
      pkl.dump(obj, f)

  @classmethod
  def load(cls, fname):
    p = Path(fname)
    with p.open("rb") as f:
      return pkl.load(f)


class Sim:

  @classmethod
  def topk(cls, dist_dict, K, return_tuple=False):
    ret = list(sorted(dist_dict.items(), key=lambda x: x[1], reverse=True))
    if return_tuple:
      return ret[:K]
    else:
      return list(zip(*ret[:K]))[0]


class Sort:

  @classmethod
  def unique_by_frequency(cls, l):
    c = Counter(l)
    sorted_list = list(zip(*c.most_common()))[0]
    sorted_list = list(sorted_list)
    return sorted_list


class Rnd:

  @classmethod
  def random_pairs(cls, l, n):
    """Randomly sample pairs without replacement from a list"""
    if len(l) * (len(l) - 1) / 2 < n:
      raise RuntimeError("Sample size greater than population")
    pairs = cls._rand_pairs(len(l), n)
    return [(l[pair[0]], l[pair[1]]) for pair in pairs]

  @classmethod
  def _decode(cls, i):
    k = math.floor((1 + math.sqrt(1 + 8 * i)) / 2)
    return k, i - k * (k - 1) // 2

  @classmethod
  def _rand_pair(cls, n):
    return cls._decode(random.randrange(n * (n - 1) // 2))

  @classmethod
  def _rand_pairs(cls, n, m):
    return [cls._decode(i) for i in random.sample(range(n * (n - 1) // 2), m)]


TempFile = tempfile.NamedTemporaryFile
