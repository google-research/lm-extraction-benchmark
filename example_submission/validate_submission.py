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

import argparse
import csv

parser = argparse.ArgumentParser("Score a solution.")

parser.add_argument("--submission", required=True)

args = parser.parse_args()

for row in csv.DictReader(open(args.submission, "r")):
    assert 'Example ID' in row
    assert 'Suffix Guess' in row
    assert 0 <= int(row['Example ID']) < 1000

    guess = list(map(int,row['Suffix Guess'][1:-1].split(",")))

    assert len(guess) == 50

    assert all(0 <= x <= 50257 for x in guess)
