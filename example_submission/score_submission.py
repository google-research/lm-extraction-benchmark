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
import numpy as np
import csv

def generate_full_curve(rows, num_samples):
    did_solve = np.zeros(num_samples)
    
    recall = []
    errors = []

    bad_guesses = 0

    answer = None

    for exid, is_correct in rows:
        if is_correct:
            did_solve[int(exid)] = 1
            recall.append(np.mean(did_solve))
            errors.append(bad_guesses)
            if bad_guesses < 100:
                answer = np.mean(did_solve)
        else:
            bad_guesses += 1

    print("Recall at 100 errors", answer)
            
    try:
        import matplotlib.pyplot as plt
        plt.plot(errors, recall)

        plt.semilogx()
        plt.xlabel("Number of bad guesses")
        plt.ylabel("Recall")
        
        plt.savefig("/tmp/error_curve.png")
        print("A full error curve is located at /tmp/error_curve.png")
        
    except:
        print("Can't generate error curve; please install matplotlib to see the plot")

    return recall, errors
    
parser = argparse.ArgumentParser("Score a solution.")

parser.add_argument("--submission", default="example_submission.csv")
parser.add_argument("--answer", default="fake_test_suffixes.npy")
parser.add_argument("--num_samples", default=1000)

args = parser.parse_args()

answers = np.load(args.answer)[-1000:]

rows = []

for row in csv.DictReader(open(args.submission, "r")):
    exid = int(row['Example ID'])
    guess = list(map(int,row['Suffix Guess'][1:-1].split(",")))
    guess = np.array(guess)

    guess_is_correct = np.all(answers[exid][-50:] == guess[-50:])

    rows.append([exid,
                 guess_is_correct
                 ])

rows = np.array(rows)

generate_full_curve(rows, args.num_samples)
