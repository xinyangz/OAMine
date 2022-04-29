import argparse
import re
import subprocess
from pathlib import Path

import utils

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpt_dir", help="Path to directories with preprocessed parsing pkl file.")
  parser.add_argument("--output_dir", help="Path to output")
  parser.add_argument("--file_pattern", default="dist-*-12.pkl", help="Pattern that will be used with glob to find pkl files.")
  parser.add_argument("--regex_pattern", default="dist-(.*)-12", help="Regex that will be used to match and capture PT names from file names.")
  args = parser.parse_args()

  cpt_dir = Path(args.cpt_dir)
  output_dir = Path(args.output_dir)

  print(f"Writing to {output_dir}")
  utils.IO.ensure_dir(output_dir)

  print(f"Searching for files w/ pattern {args.regex_pattern}")
  for mat_file in cpt_dir.glob(args.file_pattern):
    pt_name = re.match(args.regex_pattern, mat_file.stem)[1]
    subprocess.call(["python", "chunk.py", "--matrix", mat_file, "--output_dir", output_dir.as_posix(), "--output_name", f"{pt_name}"])

