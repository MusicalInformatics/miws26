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
* `melody_utils.py`: helper functions of the notebook (loading the data, piano rolls with Partitura, windows, note probabilities, threshold, graph search, and playback),
* `melody_training.py`: the training of the CNN (train/validation/test split, data augmentation, loss, and early stopping),
* `skyline.py` and `graph_animation.py`: the skyline algorithm and the animations of the notebook,
* `weights/`: the weights of the CNNs that the authors of the paper trained on the Mozart and the Pop datasets, converted to PyTorch (see the license below), and `convert_theano_weights.py`, the script that converts them.

## Teams

You work in teams of **up to 3 people**, and all the members of a team get the same grade. Select your team on the Moodle page. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

## Challenge

The challenge server, the test set, the metric and the baseline submission for this task are **to come**. They will be announced here and on Moodle.

### Dataset

The notebook uses the **Mozart dataset** of the paper: 38 movements of 13 Mozart piano sonatas, in which a professional pianist annotated the main melody line. The first time you run the notebook, it downloads the dataset (4 MB) from the companion repository of the paper (<https://github.com/LIMUNIMI/Symbolic-Melody-Identification>) to the folder `datasets/` of this folder, which is ignored by git. Each piece is a list of notes with the fields `pitch`, `onset` and `duration` (in beats), and `soprano`, which is 1 if the note is part of the melody. The notebook splits the pieces (not the windows of the piano rolls!) into 24 for training, 6 for validation and 8 for testing, as in the code of the paper.

### Pre-trained weights and licenses

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
