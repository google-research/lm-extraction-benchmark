# Training Data Extraction Challenge

Recent work has demonstrated the feasibility of training data extraction attacks
on neural language models, where an adversary can interact with a pretrained model to
recover individual examples contained in the training dataset.
For example, the GPT-2 language model memorizes the name, email address, phone
number, fax number, and physical address of a single person whose information
was contained in the model's training dataset (Carlini et al. 2021).
This presents various privacy risks, raises questions around generalization of
language models, and generally is a surprising property of model training
(Feldman 2020).

While existing attacks are strong proof-of-concepts, existing attacks aren't close
to extracting the uppper bound of what the model has memorized.
For example, out of GPT-2's entire 40 GB training dataset, just 600 examples
were shown to be extractable by Carlini et al. (2021), for a total of 0.00000015%.
However, recent work has found that large language models memorize as
much as *a few percent* of their training datasets (Carlini et al. 2022),
but current attacks are quite inefficient (Lehman et al. 2021, Kandpal et al. 2022).


## Objective

In this challenge, you will improve *targeted* data extraction attacks.
In a targeted attack, you are provided with a prefix sequence 
and must find the specific continuation (suffix) such that the entire sequence 
is contained in the training dataset.
For example, if the training dataset contains the sentence
"My phone number is 123-4567", 
and we provide you with the prefix
"My phone number is", you should output the guessed suffix "123-4567".
This differs from "untargeted" attacks, where an adversary searches for
any memorized sample, i.e.,
any data that appears somewhere in the training dataset.

Why targeted attacks?

* They matter more. It is more security critical if an adversary can
recover information related to particular topics than just be able to
recover some (arbitrary, uncontrolled) data that might be uninteresting.

* They are easier to evaluate. Evaluating an untargeted attack requires checking
if the sequence is contained *anywhere* in the training dataset;
for The Pile, this would require searching over 800GB of text.
This is challenging.
In contrast, targeted attacks only require checking if the suffix of this particular
sequence is correct.



## Dataset

Our benchmark consists of a subset of 20,000 examples contained
in [The Pile](https://arxiv.org/abs/2101.00027)'s training dataset.
This dataset has been used to train many recent large language models,
including GPT-Neo 1.3B, the model you will be extracting data from.

Each example is split into a prefix and suffix, each 50 tokens long.
Given the prefix, the task of the attack is to predict the suffix.

These 20,000 examples are designed to be somewhat easy-to-extract.
They were all chosen to meet the property that there exists a prefix length that causes the model to generate the suffix string exactly.
Moreover, we only choose examples that are *well specified*, in the sense
that given the 50-token prefix, there is only one continuation such that the entire sequence is contained in the training dataset.
Therefore, we have good reason to believe that a sufficiently strong attack should
be able to perfectly complete the suffix given each example's 50-token prefix.

We will upload three datasets.
* [datasets/train_dataset.csv](datasets/train_dataset.csv) is a set of 15,000 examples from The Pile and can be used
  to develop your attack algorithm how ever you please.
  This dataset contains both the prefixes and suffixes for your use. We recommend that you
  reserve the last 1,000 examples as an internal validation set to compute the
  efficacy of your attack, but that's up to you.
* [datasets/val_prefixes.npy](datasets/val_prefix.npy) is a set of 1,000 examples sampled from the Pile identically to
  the training set; the prefix set was released on November 14th and the suffixes will be released on December 14th.
  This will be the dataset used to run the validation round (see below for details).
* test_prefix.csv will also be a set of 1,000 examples sampled identically to the above two,
  with the prefixes released on January 23rd, 2023 and the suffixes released on January 27th,
  2023.

Unfortunately we are unable to host the exact dataset as a numpy array in this repository.
But an [unofficial repository](https://github.com/ethz-privsec/lm-extraction-benchmark-data)
maintained by our collaborator at ETH Zürich contains the data.

Alternatively, if you would like to generate the data on your own from the
original [Pile dataset](https://pile.eleuther.ai/),
we have released CSV files that contain pointers into The Pile.
The script [load_dataset.py](load_dataset.py) takes the CSV and a copy of The Pile
on your disk and will pull out the actual dataset itself, generating
`train_dataset.npy`, `train_prefix.npy` and `train_suffix.npy`.
It will also give `train_preprefix.npy` which is 100 extra tokens of context.

These examples have the property that
`np.concatenate([train_preprefix[0], train_prefix[0], train_suffix[0]]) = train_dataset[0]` is a substring of
The Pile. Your task is to use `train_prefix` to complete `train_suffix`.


## Solution Format

You will upload your solutions as an **ordered** CSV with the following format

```
    Example ID, Suffix Guess
    8,          "[3, 6, 9]"
    12,         "[4, 2, 8]"
    8,          "[3, 7, 9]"
    7,          "[1, 2, 3]"
    9,          "[0, 0, 0]"
	...
```

*Example ID* is an integer between 0 and 1000 (the total number of test samples).

*Suffix Guess* is a python-style length-50 list of the tokens in the predicted suffix.
This data should be tokenized with the GPT Neo 1.3B tokenizer, which is
identical to the GPT-2 tokenizer.

**You should order examples by how likely you believe they are to be training data**.
The first example in the list should be the suffix guess you are most confident
is correct completeion of the corresponding prefix.
The last example is the one that you are least confident is the correct suffix.

For example, in the above file, the first guess is that example 8 ends with the string
```[3, 6, 9```
the second guess is that example 12 ends with the string
```[4, 2, 8]```
and then the third guess is that the same example 8 ends with the alterante
```[3, 7, 9]```
where here we've made a different guess for the second byte.
(Obviously the sequences here should be 50 tokens long.)

There are no other constraints on this file.
You do not need to make a guess for every example.
You can make multiple guesses for some examples,
and just a a single guess (or even none at all) for other examples.
However once you have made the 100th error we will stop processing lines.

You may, if you wish, output at most 100,000 guesses and we will generate a full
precision/recall curve using these guesses. However please note because you are
allowed **at most 100** incorrect guesses and so
any rows after the 1,100th will be ignored for the competition metric.


## Evaluation Metric

There are three dimensions that determine how well an extraction attack works:
(1) the *recall* (the number of examples where you guess suffix correctly),
(2) the *precision* (how often your guesses are correct),
and (3) the *speed* (how long it takes your attack to run).
In an ideal world an attack would be evaluated across all three dimensions at the same time.
For example, the baseline attack (discussed below) shows the following curves when we
run it for varying amounts of time.


But contests need to be evaluated on a single metric, and so we had to decide on
some way to reduce an entire 3 dimensional curve to just a single point.
Ultimately we decided on measuring recall, at 100 incorrect guesses, when constrained
to 24 hours of runtime on a fixed platform.
This rewards adversaries who can perform extraction quickly and with few mistakes.

We will release exact specifications of the hardware upon the release of the validation
set, and will update this section when that happens. For the time being you can safely
assume it will be (something like) a P100 GPU with a single-digit-of-cores CPU.

We selected these thresholds after an initial evaluation of existing baseline attacks.
We found that most existing attacks saturate after >10 hours of compute, and so
by allowing 24 hours of compute we hope to limit the effect of code-level optimization
on the attack success rate.
We selected 100 total errors because this corresponds to a ~35% recall for existing
attacks, leaving significant room for improvement.


## How To Submit

You will submit to us three artifacts by emailing lm-extraction-competition@googlegroups.com:

 * A solution CSV file as described above.

 * Code that will reproduce your results when supplied with a single argument containing the prefixes to attack.

 * A short (2-5 page) description of your attack.

 * A list of team members, your institution/affiliation if you have one, and an optional team name.

We will re-run your code to verify that it meets the 24 hour runtime requirement.
Please document the steps necessary to run your code.
While we make no hard constraints on how to do this, either a conda environment, a pip requirements.txt, or a docker setup would be appreciated.
We will try to fix minimal bugs we encounter, and may even email teams if we have small challenges in reproducing results, but please test your code on a clean machine.
(You can assume CUDA drivers, whichever version you depend on, have been installed.)


## Example Submission

We have uploaded an example submission at [example_submission/example_solution.csv](example_submission/example_solution.csv)
and the code we will use to score submission at [example_submission/score_submission.py](example_submission/score_submission.py).
Internally the scoring file expects a numpy file of the suffixes [example_submission/fake_test_suffixes.npy](example_submission/fake_test_suffixes.npy).

All of these data are completely random and synthetically generated to show the format of an upload.


## Baseline Code

We have published sample baseline code to run a simple attack at [baseline/simple_baseline.py](baseline/simple_baseline.py).
This code will run for a few hours on a P100 GPU to produce a set of guessed
solution file.

```
cd baseline
python3 simple_baseline.py --root-dir tmp/ --experiment-name test/
```

If you would like to see simple results in just a few minutes you can reduce the number of guesses per example by running

```
cd baseline
python3 simple_baseline.py --root-dir tmp/ --experiment-name test/ --num-trials 1
```

By default this code runs on the **last 1000** examples of the training set,
and when it finishes will give you a few files called `guess1.csv` to `guess100.csv`
with the guesses that are being made.

From here you can then run [example_submission/score_submission.py](example_submission/score_submission.py) as follows

```
python3 ../example_submission/score_submission.py --submission guess1.csv --answer ../datasets/train_suffix.npy
```

## Cheating

Please don't!
There is no prize for winning this contest. You don't get anything out of cheating.
You will most likely be able to cheat without us noticing.

It's not very hard: our test set consists of sequences contained in The Pile---a dataset
you already need to download to extract the training sequences and test prefixes.
And so you can literally grep for the prefixes from The Pile, read off the suffixes,
and submit those as the answer. You can also search for the prefixes on the internet and
find many of them online. Please also don't do this for the test examples.

In order to mitigate the effects of cheating, we will require any participants
who wish to be ranked on the official leaderboard to send us their code **before**
the specific test set is released (see the timeline below).
We will not release your code publicly (but we encourage you to do so), and will only use
it to verify that running your code produces the claimed answers in the allocated runtime.
In extremely exceptional situation where your code needs to be
changed minimally to run on the provided test set, you may send us an update to your
code if necessary.

Querying other models trained on The Pile (other than the provided 1.3B GPT-Neo
model) is not allowed. The reasoning for this is that larger models exhibit more memorization.
Querying models trained on other datasets that do not significantly overlap with The Pile is allowed.

If you're not sure if something is against the rules, please raise an issue to ask.


## Timeline

* August 22nd: Contest announced. We release in this repository the rules, submission criteria, and training dataset.

* November 14th: Validation prefixes released. This validation round will allow participants to check how well they are doing compared to others, and also allow everyone to verify they are able to follow the submission criteria correctly.

* December 9th AOE: Validation round closes. Teams submit validation set solutions by emailing CSV file to lm-extraction-competition@googlegroups.com.

* December 14th: Validation scores announced. We will run each team's validation submission and release publicly how well everyone does. At this time we will also release the validation suffixes for teams to be able and reproduce our ranking them self.

* January 20th AOE: Final Code submission deadline. As discussed above, to mitigate cheating we will require teams submit code ahead of the test set release. This is that deadline.

* January 23th: Test prefixes released at [datasets/test_prefix.npy](datasets/test_prefix.npy). Teams will then have five days to run their code on the test prefixes---because at most 24 hours of compute are allowed for the final submission, this should be more than enough.

* January 27th AOE: Test round closes. Teams submit test set suffixes, a short 2-5 page paper describing the techniques being used, and (in exceptional circumstances) a .patch file modifying the code to run on the test set.


## Additional Details

A detailed description of the dataset construction process is available at [detailed_description.pdf](detailed_description.pdf).


## Questions

If you have any questions raise an issue on this repositories issue tracker.


## Organiziation

This contest is being run by (in alphabetical order) Nicholas Carlini, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Milad Nasr, Florian Tramer and Chiyuan Zhang.


## Legal Stuff

This is not an officially supported Google project.

## References

* Carlini, Nicholas, et al. ["Extracting Training Data from Large Language Models."](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting) *30th USENIX Security Symposium*. 2021.
* Carlini, Nicholas, et al. ["Quantifying Memorization across Neural Language Models."](https://arxiv.org/pdf/2202.07646.pdf) *arXiv preprint arXiv:2202.07646* 2022.

* Feldman, Vitaly. ["Does learning require memorization? a short tale about a long tail"](https://dl.acm.org/doi/pdf/10.1145/3357713.3384290) *STOC* 2022.

* Kandpal, Nikhil, et al. ["Deduplicating training data mitigates privacy risks in language models"](https://arxiv.org/abs/2202.06539) *ICML* 2022.

* Lehman, Eric, et al. ["Does BERT Pretrained on Clinical Notes Reveal Sensitive Data?"](https://arxiv.org/abs/2104.07762). *NAACL* 2021.

## Presented Submissions
These submissions were presented during the [SATML conference:](https://www.youtube.com/watch?v=IPs7BSNmy5A)

* Yu, Weichen, et al. ["Bag of Tricks for Training Data Extraction from Language Models"](https://arxiv.org/abs/2302.04460)
* Al-Kaswan, Ali, et al. ["Targeted Attack on GPT-Neo for the SATML Language Model Data Extraction Challenge"](https://arxiv.org/abs/2302.07735)
