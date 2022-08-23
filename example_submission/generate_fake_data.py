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

import numpy as np
import csv

np.random.seed(0)

suffix = np.random.randint(0, 10, size=(1000, 10))
hardness = np.random.uniform(size=1000)

np.save("fake_test_suffixes.npy", suffix)

out = csv.writer(open("example_submission.csv", "w"))
out.writerow(["Example ID", "Suffix Guess"])

rows = []

for _ in range(10):
    for i in range(1000):
        pr = np.random.uniform(0, 1)
        g = 1/(1+np.exp(-(-pr*2+1)*4))
        if np.random.uniform() * hardness[i] > g:
            guess = suffix[i,:]
        else:
            guess = np.random.randint(0, 10, size=10)
        rows.append([i, pr, str(list(guess)).replace(" ", "")])

order = np.argsort([-x[1] for x in rows])
        
rows = [(rows[i][0], rows[i][2]) for i in order]

for r in rows:
    out.writerow(r)
