# Melody Identification Task

Melody identification is one of the **Symbolic MIR Challenges** of the course: you identify which notes of a score belong to the melody, i.e., the most salient line of the piece, and which ones are accompaniment.

This task is worth **100 points** (report: 75%, final presentation: 25%).

| Deadline | Date |
|:---|:---|
| Report and submission to the challenge server | **December 31, 2026, 23:59** |
| Final presentations (attendance is mandatory) | **January 13 or January 20, 2027** |

The folder contains

* `melody_identification.ipynb`: notebook with an introduction to melody identification with a convolutional neural network (CNN) on piano rolls, following the paper of Simonetta et al. (2019). It covers the piano roll representation, the skyline algorithm (a simple baseline), the CNN, the steps to obtain a melody line from the output of the CNN (note probabilities, threshold, and a graph search), and the training of the CNN,
* `melody_cnn.py`: the CNN, implemented in PyTorch,
* `melody_utils.py`: helper functions of the notebook (loading the data, piano rolls with Partitura, windows, note probabilities, threshold, graph search, the melody of a piece with the CNN, and playback),
* `melody_training.py`: the training of the CNN (train/validation/test split, data augmentation, loss, and early stopping),
* `skyline.py` and `graph_animation.py`: the skyline algorithm and the animations of the notebook,
* `weights/`: the weights of the CNNs that the authors of the paper trained on the Mozart and the Pop datasets, converted to PyTorch (see the license below), and `convert_theano_weights.py`, the script that converts them. It also has the weights of a CNN that we trained from scratch (`train_melody_id.py`),
* `Baseline_MelodyIdentification.py`: the baseline submission (the CNN, trained from scratch, with the steps of the paper),
* `melody_identification_challenge.py`: the code that evaluates the submissions (the F1-score of the melody notes).

## Teams

You work in teams of **up to 3 people**, and all the members of a team get the same grade. Select your team on the Moodle page. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

## Challenge

For this challenge, you have to predict, for each note of the pieces of a test set, whether it is part of the melody, and export your results as a compressed numpy file. You can use **any method that you want** (even if it is not the skyline algorithm or the CNN of the notebook), but your implementation **must use the Partitura library**.

* Your team has to submit to the challenge server **at least once**. The challenge server, with the leaderboard, is at <https://challenges.cp.jku.at/challenge/42/>.
* There is a leaderboard. Your position on the leaderboard **does not influence your grade**, but the winners of the challenge receive a small prize!
* If you chose other challenges for your report, you can submit to this one as an extra: you get 20 extra points if you beat the baseline. In this case you do not have to write a report. You only submit your results to the challenge server and upload your code.

### Dataset

The notebook uses the **Mozart dataset** of the paper: 38 movements of 13 Mozart piano sonatas, in which a professional pianist annotated the main melody line. You do **not** download the original dataset: the notebook works with the training set of the challenge (below), which you get from Moodle and unzip in the folder `datasets/melody_id_challenge` of this folder (ignored by git). The notebook splits the 29 pieces of the training set by sonata (not the windows of the piano rolls!) into 23 for training and 6 for validation. The 9 pieces of the test set are only evaluated on the challenge server.

For the challenge, the pieces are split by sonata (as in the notebook) and written as csv files. Download the zip file of the dataset from Moodle and unzip it (the folder `datasets/` of this folder is ignored by git, so it is a good place for it). It contains two folders:

* `train`: 29 pieces **with the annotations** of the melody. The name of a file is the Köchel number of the sonata and the movement (e.g., `kv279_1.csv`), so that you can keep the movements of the same sonata together when you split the training set for validation. Each file has one row for each note, with the columns `id` (unique id of the note in the piece), `onset_beat` and `duration_beat` (in beats), `pitch` (MIDI pitch) and `melody` (1 if the note is part of the melody, 0 otherwise). The notes are sorted by onset and then by pitch, and the grace notes (notes with no duration) are not included. The file `melody_identification_train_gt.npz` has the annotations of all the pieces in the format of the submissions (see below).
* `test`: the 9 movements of the sonatas K. 280, 332 and 457, **without the annotations** (the columns are the same, but there is no `melody` column). The files have anonymous names (`test_000.csv`, ...).

The test set is the test set of the notebook, and the training set is the union of its training and validation sets. **Be careful with the weights of the authors** (`weights/melody_cnn_mozart.pt`): they have seen the test set of the challenge (see the warning in the section on the pre-trained weights below). The weights `melody_cnn_mozart_reproduced.pt` were trained from scratch with the training pieces of the notebook (`train_melody_id.py`), and **not** with the test pieces.

In the following, the location where you unzip the file is called `path_to_the_data_directory`.

### Metric

The submissions are evaluated with the **F1-score of the melody notes** (the higher, the better). For each piece of the test set, we compute the precision and the recall of the notes that are predicted to be melody, and their harmonic mean (the F1-score). The score is the **mean of the F1-scores of the pieces**, so that all the pieces have the same weight, whatever their length. It is a number between 0 and 1. The notes that are not in the submission (an array that is too short) are considered as not melody, and a piece that is not in the submission has an F1-score of 0. The implementation is in `melody_identification_challenge.py`.

### Format of the submission

The submission is a **compressed numpy file** (`.npz`) with an array for each file of the test set. The name of the array is the name of the file without the extension (`test_000`, ...), and the array has one value for each note of the file, **in the order of the rows of the file**: 1 if the note is predicted to be part of the melody, and 0 otherwise.

```python
import numpy as np

predictions = {
    "test_000": np.array([0, 1, 1, 0, ...]),  # one value for each row of test_000.csv
    "test_001": np.array([...]),
    ...
}
np.savez_compressed("results_melody_identification.npz", **predictions)
```

The file `melody_identification_train_gt.npz` of the training set has the annotations in the same format (an array for each file, named as the file without the extension), so you can evaluate your method locally (for this, your file must have the predictions of the pieces of the training set):

```bash
python melody_identification_challenge.py --submission your_results.npz --target path_to_the_data_directory/train/melody_identification_train_gt.npz
```

The score is printed in the terminal.

### Running the baseline

The baseline is the CNN of the notebook, with the weights trained from scratch on the training set (`melody_cnn_mozart_reproduced.pt`), and the steps of the paper to get a melody line (probability of each note, threshold of the clustering, and shortest path in the graph of the notes). To run the baseline submission, run the following commands on the terminal:

```bash
cd PATH_TO_THE_REPO/symbolic_tasks_II/melody_id
conda activate miws26
python Baseline_MelodyIdentification.py -i path_to_the_data_directory -o baseline_results_melody_identification.npz
```

This evaluates the method on the training set (using the labels; since the CNN was trained on these pieces, the result is optimistic), and generates the file with the predictions for the test set. The baseline reaches a mean F1-score of 0.898 on the training set and **0.847 on the test set**. Upload the npz file to the challenge server to see the score of the baseline.

### Your submission

Your script should be executed in the following way:

```bash
python TeamName_MelodyIdentification.py -i path_to_the_data_directory -o results_melody_identification.npz
```

This generates a file with your results (`results_melody_identification.npz`, you can choose whichever name you want). Please upload this file to the [challenge server](https://challenges.cp.jku.at/challenge/42/).

The file `TeamName_MelodyIdentification.py` should be as self-contained as possible, and you can use third-party libraries that are not included in the conda environment of the course. Please follow the example in `Baseline_MelodyIdentification.py`.

### Pre-trained weights and licenses

> [!WARNING]
> **The weights of the authors of the paper for the Mozart dataset (`weights/melody_cnn_mozart.pt`) have seen the test set of the challenge.** The authors trained this CNN with **all** the pieces of the Mozart dataset, and the 9 pieces of the test set of the challenge (the sonatas K. 280, 332 and 457) are part of it. Using these weights to predict the test set is therefore a **data leak**: the score is not a fair evaluation of the method (see the criteria of the evaluation in the grading of the report). If you use them, you have to say so in the report, and you should not present the score on the test set as the performance of your method. The weights `melody_cnn_mozart_reproduced.pt` were trained from scratch with the training pieces only (`train_melody_id.py`), so they are safe to use. The weights for the Pop dataset (`melody_cnn_pop.pt`) were not trained with Mozart pieces.

The weights in `weights/` are the ones released by the authors of the paper under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license, converted to PyTorch (see `weights/LICENSE.md`). If you use them, you have to give credit to the authors. The code of this folder is adapted from the code of the paper, which is released under the MIT license (see `LICENSE.md` for the license and for how we adapted the code). The reference is:

> Federico Simonetta, Carlos Cancino-Chacón, Stavros Ntalampiras and Gerhard Widmer (2019). "A Convolutional Approach to Melody Line Identification in Symbolic Scores". In *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, Delft, The Netherlands.

## Report

For the project you have to submit a report. **Only complete submissions will be graded.** Upload a **zip file** (max. 500 MB) to Moodle with the following four parts:

1. **The report as a pdf file**, using the ISMIR template (**4 pages** of content, plus additional pages only for references). The report must **explicitly state if and how you used LLMs** for the task.
2. **The Python scripts** used for the challenge (including `TeamName_MelodyIdentification.py`, any other helper files, and trained model weights, if relevant). Your implementation **must use** the Partitura library.
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
    * Why did you select your approach (how does the method address the particular musical problem)?
    * A brief description of the method(s), and the parameters of the method (and how they relate to the problem). You don't need to include a full technical description of the methods!

3. **Evaluation of the method(s)**
    * How do you evaluate the performance of the method? Which metrics do you use? Include both your own evaluation using the training set, and the performance of the method on the leaderboard.
    * The loss function used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the cross entropy, but the metric used to compare the models could be the F1-score of the melody notes).
    * Which datasets do you use, what information do they contain, and which features do you use?
    * How was the method trained (including the strategies for hyperparameter selection)?

4. **Discussion of the results and your own conclusions**
    * What worked and what did not? Which characteristics of the method lead to better performance?
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
