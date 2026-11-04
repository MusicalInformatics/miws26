# Key Estimation Task

Key estimation is one of the **Symbolic MIR Challenges** of the course: you identify the main tonality (the key) of a piece from MIDI-like note information.

This task is worth **100 points** (report: 75%, final presentation: 25%).

| Deadline | Date |
|:---|:---|
| Report and submission to the challenge server | **December 31, 2026, 23:59** |
| Final presentations (attendance is mandatory) | **January 13 or January 20, 2027** |

The folder contains

* `key_estimation.ipynb`: notebook with an introduction to key estimation (key profiles, the Krumhansl-Schmuckler method, and the tonal distance between keys),
* `Baseline_KeyEstimation.py`: the baseline submission (Krumhansl-Schmuckler),
* `key_estimation_challenge.py`: the code that evaluates the submissions (the tonal distance),
* `key_profiles.py`, `key_estimation_utils.py` and `example_data/`: helper code and examples for the notebook.

## Teams

You work in teams of **up to 3 people**, and all the members of a team get the same grade. Select your team on the Moodle page. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

## Challenge

For this challenge, you have to predict the key (i.e., the tonality) of each MIDI file of a test set, and export your results as a csv file. You can use **any method that you want** (even if it is not one of the methods presented in the lecture), but your implementation **must use the Partitura library**.

* Your team has to submit to the challenge server **at least once**. The server is at <https://challenges.cp.jku.at/challenge/38/>.
* There is a leaderboard. Your position on the leaderboard **does not influence your grade**, but the winners of the challenge receive a small prize!
* If you chose other challenges for your report, you can submit to this one as an extra: you get 20 extra points if you beat the baseline. In this case you do not have to write a report. You only submit your results to the challenge server and upload your code.

### Dataset

Download the dataset from Moodle and unzip it (the folder `key_estimation_dataset` of this folder is ignored by git, so it is a good place for it). It contains two folders:

* `train`: MIDI files and a csv file (`key_estimation_train_gt.csv`) with the key of each file (columns `filename` and `key`, e.g., `C`, `G#m`),
* `test`: only MIDI files (there are no labels).

In the following, the location where you unzip the file is called `path_to_the_data_directory`.

### Metric

The submissions are evaluated with the **average tonal distance** between the predicted and the true key of each file in the test set (the lower, the better). The tonal distance measures how close two keys are in the circle of fifths. It is the number of steps between the two keys in the circle of fifths (from 0 to 6), plus 1 if one key is major and the other one is minor. For example, the distance between C major and its relative minor, A minor, is 1, and the distance between C major and F# major is 6. Enharmonically equivalent spellings of a key (e.g., `A#` and `Bb`) are considered the same key. Files for which there is no prediction get the maximal distance (7). The implementation is in `key_estimation_challenge.py`, and the notebook `key_estimation.ipynb` explains it.

The baseline script also prints the accuracy and the macro F1-score on the training set. These are useful for your own evaluation, but the score of the challenge is the tonal distance.

### Running the baseline

To run the baseline submission, run the following commands on the terminal:

```bash
cd PATH_TO_THE_REPO/symbolic_tasks_I/1_key_estimation
conda activate miws26
python Baseline_KeyEstimation.py -i path_to_the_data_directory -o baseline_results_key_estimation.csv
```

This evaluates the method on the training set (using the labels), and generates the csv file with the predictions for the test set. The csv file has a header, and one line for each file of the test set with the name of the file and the predicted key:

```
filename,key
mi2024_key_estimation_test_125.mid,C
mi2024_key_estimation_test_119.mid,F
...
```

Upload this csv file to the challenge server to see the score of the baseline.

### Your submission

Your script should be executed in the following way:

```bash
python TeamName_KeyEstimation.py -i path_to_the_data_directory -o results_key_estimation.csv
```

This generates a csv file with your results (`results_key_estimation.csv`, you can choose whichever name you want). Please upload this file to the [challenge server](https://challenges.cp.jku.at/challenge/38/).

The file `TeamName_KeyEstimation.py` should be as self-contained as possible, and you can use third-party libraries that are not included in the conda environment of the course. Please follow the example in `Baseline_KeyEstimation.py`.

## Report

For the project you have to submit a report. **Only complete submissions will be graded.** Upload a **zip file** (max. 500 MB) to Moodle with the following four parts:

1. **The report as a pdf file**, using the ISMIR template (**4 pages** of content, plus additional pages only for references). The report must **explicitly state if and how you used LLMs** for the task.
2. **The Python scripts** used for the challenge (including `TeamName_KeyEstimation.py`, any other helper files, and trained model weights, if relevant). Your implementation **must use** the Partitura library.
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
    * The loss function used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the cross entropy, but the metric used to compare the models could be the accuracy, the F1-score or, in this task, the tonal distance).
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
