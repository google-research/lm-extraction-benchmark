# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Builds the .numpy files of token sequences used for analyzing memorization.

Example usage:
PILE_DIR="/home/ncarlini/pile/the-eye.eu/public/AI/pile/train/"
python3 load_dataset.py $PILE_DIR train

"""

import os
import csv
import pickle
import multiprocessing as mp
import numpy as np
import json
from transformers import GPT2Tokenizer
import sys
import hashlib

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encoder(args):
  exid, next_line, start_byte, end_byte, count = args
  encoded = tokenizer.encode(next_line)
  sequence = encoded[start_byte:end_byte]
  return exid, sequence, count

if __name__ == "__main__":

  if len(sys.argv) != 3:
    print("USAGE: python build_pile_dataset.py $PILE_DIR $SPLIT")
    exit()

  pile_path = sys.argv[1]
  output_path = "datasets"
  split = sys.argv[2]

  pile_files = [open(pile_path+"%02d.jsonl"%x) for x in range(30)]

  ds = {
    'train': '6d85d7f2b48e34e6f96541a546a699bf',
    'val': '318d6f3d68dfd3497956a79678067c62',
    'test': 'ddcc8318f0cf1f857a6adcb18e6726b8',
    'rest': 'f81027fe935c2260ae34a82e0e8c5434',
  }
  
  assert split in ds
  
  try:
    fin = open("datasets/"+split+"_dataset.csv")
  except:
    print("The split", split, "does not exist (yet?).")
    exit(1)

  prompts = {}
  counts = {}
  

  # Load the examples indicated by the byte offsets in the scaling dataset csv.
  for i,row in enumerate(csv.reader(fin)):
    if i%1000 == 0:
      print(i)
    exid, fid, line_byte_offset, start, end, take_before, take_after, internal_offset, size, start_byte, end_byte, count = map(int,row)

    pile_files[fid].seek(line_byte_offset)
    next_line = json.loads(next(pile_files[fid]))['text']

    if start_byte < 0:
        # Faaaast!
        # Here be dragons...
        next_line = bytes(next_line, 'utf8')
        sequence = tokenizer.encode(next_line[start - take_before:end + take_after].decode("utf8", "ignore"))[internal_offset:internal_offset+size]
        if len(sequence) == 0:
            sequence = tokenizer.encode("z"+next_line[start:end + take_after].decode("utf8", "ignore"))[1:size+1]
    else:
        encoded = tokenizer.encode(next_line)
        sequence = encoded[start_byte:end_byte]

    if len(sequence) > 0:
      prompts[exid] = sequence
      counts[exid] = count
    else:
      print("PASS", i)

  if not os.path.exists(output_path):
      os.mkdir(output_path)
  prompts = [x[1] for x in sorted(prompts.items())]
  prompts = np.array(prompts, dtype=np.uint16)

  print(hashlib.md5(prompts.tobytes()).hexdigest())
  assert hashlib.md5(prompts.tobytes()).hexdigest() == ds[split]
  
  np.save(os.path.join(output_path, split+"_dataset.npy"), prompts)

  np.save(os.path.join(output_path, split+"_preprefix.npy"), prompts[:, :100])
  np.save(os.path.join(output_path, split+"_prefix.npy"), prompts[:, 100:150])
  np.save(os.path.join(output_path, split+"_suffix.npy"), prompts[:, 150:200])
  
  
  
