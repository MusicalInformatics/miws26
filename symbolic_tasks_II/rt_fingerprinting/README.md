# Real-Time Piece Fingerprinting Task

Real-time piece fingerprinting is one of the **Symbolic MIR Challenges** of the course: a pianist plays and switches from one piece to another, and your system has to tell, from the notes that it has heard so far, which piece is being played. This is the problem behind score following systems that do not know in advance what the pianist is going to play (for example, when the pianist improvises by moving between pieces).

This task is worth **100 points** (report: 75%, final presentation: 25%).

| Deadline | Date |
|:---|:---|
| Report and submission to the challenge server | **December 31, 2026, 23:59** |
| Final presentations (attendance is mandatory) | **January 13 or January 20, 2027** |

The folder contains

* `fingerprinting.ipynb`: notebook with an introduction to the task and to symbolic fingerprinting (the data, the tokens made of three notes, the hash table of the scores, the identification of the piece with the histograms of the time differences, and the evaluation), following the paper of Arzt et al. (2012),
* `fingerprinting.py`: the tokens, the hash tables, and the online systems (`OnlineFingerprinting`: histogram bins, as in the paper, and `OnlineFingerprintingVoting`: a simpler system that counts votes),
* `fingerprinting_plots.py`: the plots, the animation and the audio of the notebook,
* `utils.py`: helper functions (loading scores and performances as Partitura note arrays, and unpacking the dataset),
* `challenge.py`: the evaluation of the notebook, with the metrics of Table 3 of the paper (piece and score position, for several query lengths). It is useful for your own experiments,
* `Baseline_Fingerprinting.py`: the baseline submission (`OnlineFingerprinting`),
* `fingerprinting_challenge.py`: the code that evaluates the submissions (the accuracy of the piece identification).

## Teams

You work in teams of **up to 3 people**, and all the members of a team get the same grade. Select your team on the Moodle page. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

## Challenge

For this challenge, you have to identify the piece that is being played in each **query** of the files of a test set, and export your results as a compressed numpy file. You can use **any method that you want** (even if it is not the fingerprinting of the baseline), but your implementation **must use the Partitura library**.

* Your team has to submit to the challenge server **at least once**. The challenge server, with the leaderboard, is at <https://challenges.cp.jku.at/challenge/41/>.
* There is a leaderboard. Your position on the leaderboard **does not influence your grade**, but the winners of the challenge receive a small prize!
* If you chose other challenges for your report, you can submit to this one as an extra: you get 20 extra points if you beat the baseline. In this case you do not have to write a report. You only submit your results to the challenge server and upload your code.

### The task

Each file is a **sequence** that simulates a pianist who plays 5 to 10 segments of different pieces, one after the other, without stopping between them (the sequences are made by concatenating slices of real performances). The pieces are the ones of the folder `scores` of the dataset (see below). The sequence is processed as a stream of notes: **the identification must be causal, as in a real-time system.** The prediction for a query can only use the notes up to the end of the query (and the ones before), and never the notes that come later. This is not checked by the server, so please respect it, and explain how your method does it in the report.

The note array of a sequence (the notes of the MIDI file, sorted by onset and then by pitch, as in `utils.load_note_array`) is split into consecutive windows of **50 notes without overlap**. These windows are the **queries**:

* the query `k` consists of the notes `50 k` to `50 k + 49` (the first query is `k = 0`). Only complete windows are queries, so a last window with less than 50 notes is not a query,
* the correct piece of a query is the piece that is being played when its **last** note is played. Note that a query can contain notes of two pieces if the pianist switches in the middle of it. In that case, the correct piece is the new one.

### Dataset

Download the zip file of the dataset from Moodle. The notebook unpacks it (`utils.unpack_data`) in the folder `data` of this folder, which is ignored by git, so it is a good place for it. It contains the following folders:

* `scores`: the MusicXML scores of the **85 pieces**, and `scores.csv` (columns `piece`, `file` and `unfold`). The name of a piece is the name of its score file without the extension (e.g., `Mozart_K331_1st-mov`), and this is what you have to predict. The pieces are: 4 pieces of the Vienna 4x22 corpus, 35 movements of Mozart piano sonatas, and 46 movements of Beethoven piano sonatas. For some scores (`unfold` = 1), the repeats are written out when the pianist plays them. For these scores, you have to **unfold the repeats** (e.g., with `partitura.score.unfold_part_maximal`) to get all the notes that are played. `utils.load_note_array(path, unfold=True)` does this.
* `train`: 150 sequences with the **ground truth**. The files are named `mi2026_fingerprinting_train_XXX.mid`, and each one has
    * `mi2026_fingerprinting_train_XXX_segments.csv`: one row for each segment of the sequence, with its `start` and `end` (in seconds), the `piecename`, the `performance` and the position `source_start` of the performance from which it comes, the `transposition` (in semitones), the estimated `key` (of the original) and the `mode_change` (`none` or e.g. `major>minor(harmonic)`, see below),
    * `mi2026_fingerprinting_train_XXX_notes.csv`: one row for each note of the sequence (`id` is the id of the note when the MIDI file is loaded with Partitura), with the `piecename` and `performance` from which it comes, the id of the note in that performance (`perf_id`), the id of the note of the score that it is aligned to (`score_id`; empty for notes that are not in the score), the `segment` and `clipped` (1 if the note started before the beginning of the segment, so its onset is not real). With this you can build, for example, the position in the score of every note,
    * and the file `fingerprinting_train_gt.npz` with the correct piece of each query of all the files of the training set (see the format below).
* `test`: 50 sequences, only the MIDI files (`mi2026_fingerprinting_test_XXX.mid`). There are no labels.

The segments are slices of 5 to 30 seconds of **real performances** of the pieces (the performances of different pianists, aligned to the scores). The challenge is **harder than only recognizing a piece**, because the pianist can **change the key and the mode** of a segment. Each segment is, with a probability of 0.4, transposed by a random number of semitones (from -6 to +6), and the mode can change (a major piece can be played in minor, and a minor piece in major). The notes of the scale that are different (the third, and sometimes the sixth and the seventh) are changed, and the chromatic notes are not. The segments that are not transformed (a probability of 0.6) are played as in the original. Think of how this changes the information that you can use to identify a piece, and how your method can deal with it. The `transposition` and `mode_change` columns of the files of the training set tell you what was done in each segment.

In the following, the location where you unzip the file is called `path_to_the_data_directory`.

### Metric

The submissions are evaluated with the **accuracy of the piece identification of the queries of 50 notes**, over all the queries of the files of the test set (the higher, the better). It is the fraction of the queries for which the predicted piece is the correct one. The score does not consider the position in the score. A file or a query without a prediction counts as wrong. The implementation is in `fingerprinting_challenge.py`.

Note that no method can be perfect, since the queries in which the pianist switches from one piece to another contain notes of two pieces.

### Format of the submission

The submission is a **compressed numpy file** (`.npz`) with an array for each file of the test set. The name of the array is the name of the file without the extension, and the array contains the predicted piece (a string, the name of the piece) of each query of the file, in order (use an empty string if you do not have a prediction):

```python
import numpy as np

predictions = {
    "mi2026_fingerprinting_test_000": np.array(["Mozart_K331_1st-mov", "Beethoven_Sonata_8-1", ...]),
    "mi2026_fingerprinting_test_001": np.array([...]),
    ...
}
np.savez_compressed("results_fingerprinting.npz", **predictions)
```

The files of the training set have the ground truth in the same format (`fingerprinting_train_gt.npz`), so you can evaluate your method locally:

```bash
python fingerprinting_challenge.py --submission your_results.npz --target path_to_the_data_directory/train/fingerprinting_train_gt.npz
```

(For this, your file must have the predictions of the files of the training set.) The score is printed in the terminal.

### Running the baseline

The baseline builds the hash table of the 85 scores (about one minute), and identifies the piece of every query with `OnlineFingerprinting` (the histogram bins), with the tokens of three notes (`n1 = n2 = 5`) of the pitches of the notes. To run the baseline submission, run the following commands on the terminal:

```bash
cd PATH_TO_THE_REPO/symbolic_tasks_II/rt_fingerprinting
conda activate miws26
python Baseline_Fingerprinting.py -i path_to_the_data_directory -o baseline_results_fingerprinting.npz
```

This evaluates the method on the training set (using the labels), and generates the file with the predictions for the test set. The baseline reaches an accuracy of 56.3% on the training set and 55.1% on the test set. Upload the npz file to the [challenge server](https://challenges.cp.jku.at/challenge/41/) to see the score of the baseline.

### Your submission

Your script should be executed in the following way:

```bash
python TeamName_Fingerprinting.py -i path_to_the_data_directory -o results_fingerprinting.npz
```

This generates a file with your results (`results_fingerprinting.npz`, you can choose whichever name you want). Please upload this file to the [challenge server](https://challenges.cp.jku.at/challenge/41/).

The file `TeamName_Fingerprinting.py` should be as self-contained as possible, and you can use third-party libraries that are not included in the conda environment of the course. Please follow the example in `Baseline_Fingerprinting.py`.

## Report

For the project you have to submit a report. **Only complete submissions will be graded.** Upload a **zip file** (max. 500 MB) to Moodle with the following four parts:

1. **The report as a pdf file**, using the ISMIR template (**4 pages** of content, plus additional pages only for references). The report must **explicitly state if and how you used LLMs** for the task.
2. **The Python scripts** used for the challenge (including `TeamName_Fingerprinting.py`, any other helper files, and trained model weights, if relevant). Your implementation **must use** the Partitura library.
3. **The conda environment YAML file** to make your code reproducible.
4. **A `README.md` file** documenting how to set up the environment and how to run the code.

The deadline is **December 31, 2026, 23:59**.

### Report template

We use a copy of the ISMIR 2026 template, adapted for the class, in Overleaf: [Overleaf template](https://www.overleaf.com/project/6ae1b6cabf7ed6ff048bbea5/share#68a3851d3fabcfb42dcde0901745a5cf860657b3f5ecb175). To use it, create your own copy of the project under *File > Make a copy*.

### Contents of the report

You can structure the report in any way you want, as long as it covers the following points:

1. **Introduction**
    * What is the task about? Why is it an interesting problem (think of musical and technical issues)?

2. **Description of the method/methods used**
    * Why did you select your approach (how does the method address the particular musical problem)? How does it deal with the changes of key and mode, and how is it causal?
    * A brief description of the method(s), and the parameters of the method (and how they relate to the problem). You don't need to include a full technical description of the methods!

3. **Evaluation of the method(s)**
    * How do you evaluate the performance of the method? Which metrics do you use? Include both your own evaluation using the training set, and the performance of the method on the leaderboard.
    * The loss function used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the cross entropy, but the metric used to compare the models could be the accuracy, or, in this task, the accuracy of the identification of the piece of queries of 50 notes). You can also evaluate your method on other query lengths, and on the position in the score (using the note maps of the training set), as in Table 3 of the paper.
    * Which datasets do you use, what information do they contain, and which features do you use?
    * How was the method trained or tuned (including the strategies for hyperparameter selection)?

4. **Discussion of the results and your own conclusions**
    * What worked and what did not? Which characteristics of the method lead to better performance? How does the performance change in the segments with and without transposition or change of mode?
    * Do not be afraid to conduct ablation studies to see how the different parts of the method contribute to the overall performance!

### Grading

The report is graded on the following aspects:

1. **Technical correctness (methods/models): 25%**
    * The methods are appropriate for the problem.
    * The methods were correctly applied/used.
    * The implementation is correct.

2. **Technical correctness (evaluation): 25%**
    * An appropriate evaluation was conducted.
    * The metrics/statistics are used correctly.
    * There is no data leak (correct split of the data), if the models are trained.

3. **Presentation style (structure, clarity): 40%**
    * Ideally, the report should not be only text or code: it should also include figures illustrating the methods and the results. Imagine that you are writing for **non-experts**, and try to explain the methods in a simple and clear way.
    * Is the report reasonably structured into sections (e.g., introduction, methods, datasets, etc.)?
    * Are the citations correctly formatted?

4. **Critical reflection: 10%**
    * What are the limitations of the approach? What did you learn?

## Final presentation

The presentation is 25% of the grade of the task, and attendance is **mandatory** for **everyone in the team** (in person or remote):

* You will present in one of the two final sessions, on **January 13** or **January 20, 2027**. Register your team for your preferred session (the slots are limited and are assigned on a first-come, first-served basis).
* Just attending the session is not enough: **everyone in the team has to participate** in the presentation. Otherwise you will **not get a grade** (the grades of the reports and extra assignments only count if you present).
* An alternative session can be organized only if you communicate it in advance (by **January 6, 2027**). In case of emergencies, a certificate is required (an exam is not considered an emergency).

## References

> Andreas Arzt, Sebastian Böck and Gerhard Widmer (2012). "Fast Identification of Piece and Score Position via Symbolic Fingerprinting". In *Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR)*, Porto, Portugal.
