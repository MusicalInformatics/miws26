# Piece Difficulty Estimation Task

Piece difficulty estimation is one of the **Symbolic MIR Challenges** of the course: you predict how difficult it is to play a piano piece from its score.

This task is worth **100 points** (report: 75%, final presentation: 25%).

| Deadline | Date |
|:---|:---|
| Report and submission to the challenge server | **December 31, 2026, 23:59** |
| Final presentations (attendance is mandatory) | **January 13 or January 20, 2027** |

The folder contains

* `piece_difficulty_estimation.ipynb`: notebook with an introduction to piece difficulty estimation with "classical" machine learning methods (features of the scores, and several classifiers),
* `Baseline_PieceDifficulty.py`: the baseline submission,
* `difficulty_estimation_challenge.py`: the code that evaluates the submissions (the macro F1-score),
* `difficulty_classification_training.csv` and `difficulty_classification_test_no_labels.csv`: the labels of the training set, and the list of the files of the test set.

## Teams

You work in teams of **up to 3 people**, and all the members of a team get the same grade. Select your team on the Moodle page. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

## Challenge

For this challenge, you have to predict the difficulty of each score of a test set, according to the [Henle difficulty scale](https://www.henle.de/us/levels-of-difficulty/) (from 1, the easiest, to 9, the most difficult), and export your results as a csv file. You can use **any method that you want** (even if it is not one of the methods presented in the lecture), but your implementation **must use the Partitura library**.

* Your team has to submit to the challenge server **at least once**. The server is at <https://challenges.cp.jku.at/challenge/39/>.
* There is a leaderboard. Your position on the leaderboard **does not influence your grade**, but the winners of the challenge receive a small prize!
* If you chose other challenges for your report, you can submit to this one as an extra: you get 20 extra points if you beat the baseline. In this case you do not have to write a report. You only submit your results to the challenge server and upload your code.

### Dataset

Download the file `Dataset.zip` from Moodle. It contains

* a folder `scores_difficulty_estimation` (in a zip file) with the scores (MusicXML files) of the training and the test sets,
* `difficulty_classification_training.csv`: the difficulty level (from 1 to 9) of each of the 369 scores of the training set (columns `file` and `difficulty`),
* `difficulty_classification_test_no_labels.csv`: the 93 scores of the test set (column `file`). There are no labels.

The classes are imbalanced (there are many more pieces of levels 5 to 7 than of level 1 or 9). The two csv files are also in this folder. If you use the notebook, you do not have to unzip anything: if the scores are not in the folder `datasets/scores_difficulty_estimation` of this folder, the notebook sets them up from `Dataset.zip` (put the file in this folder, or, in Google Colab, upload it when you are asked to). The location of the folder with the scores will be referred to as `path_to_the_scores_difficulty_estimation` in the following.

### Metric

The submissions are evaluated with the **macro F1-score** between the predicted and the true difficulty levels of the scores in the test set (the higher, the better). The macro F1-score is the average of the F1-scores of each level, so all the levels are equally important, no matter how many pieces they have. Files for which there is no prediction count as a wrong prediction. The implementation is in `difficulty_estimation_challenge.py`.

The baseline script also prints the accuracy and the macro F1-score on a validation set. These are useful for your own evaluation, but the score of the challenge is the macro F1-score on the test set.

### Running the baseline

To run the baseline submission, run the following commands on the terminal (from this folder, since the script reads the csv files of this folder):

```bash
cd PATH_TO_THE_REPO/symbolic_tasks_I/2_piece_difficulty
conda activate miws26
python Baseline_PieceDifficulty.py -i path_to_the_scores_difficulty_estimation -o baseline_results_difficulty_estimation.csv
```

This trains a decision tree on a few features of the scores of the training set, and generates the csv file with the predictions for the test set. The csv file has a header, and one line for each file of the test set with the name of the file and the predicted level:

```
file,difficulty
3.musicxml,5
K576-3.musicxml,6
...
```

Upload this csv file to the challenge server to see the score of the baseline.

### Your submission

Your script should be executed in the following way:

```bash
python TeamName_PieceDifficulty.py -i path_to_the_scores_difficulty_estimation -o results_difficulty_estimation.csv
```

This generates a csv file with your results (`results_difficulty_estimation.csv`, you can choose whichever name you want). Please upload this file to the [challenge server](https://challenges.cp.jku.at/challenge/39/).

The file `TeamName_PieceDifficulty.py` should be as self-contained as possible, and you can use third-party libraries that are not included in the conda environment of the course. Please follow the example in `Baseline_PieceDifficulty.py`.

## Report

For the project you have to submit a report. **Only complete submissions will be graded.** Upload a **zip file** (max. 500 MB) to Moodle with the following four parts:

1. **The report as a pdf file**, using the ISMIR template (**4 pages** of content, plus additional pages only for references). The report must **explicitly state if and how you used LLMs** for the task.
2. **The Python scripts** used for the challenge (including `TeamName_PieceDifficulty.py`, any other helper files, and trained model weights, if relevant). Your implementation **must use** the Partitura library.
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
    * The loss function used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the cross entropy, but the metric used to compare the models could be the accuracy or the F1-score).
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
