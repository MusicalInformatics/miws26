# License and attribution of the code

The code in this folder is adapted from the code of the paper

> Federico Simonetta, Carlos Cancino-Chacón, Stavros Ntalampiras and Gerhard Widmer (2019). "A Convolutional Approach to Melody Line Identification in Symbolic Scores". In *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, Delft, The Netherlands.

available at <https://github.com/LIMUNIMI/Symbolic-Melody-Identification> (we used the commit `ecf8833`), which is released under the MIT license:

```
MIT License

Copyright (c) 2019 symbolicmelodyidentification

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The pre-trained weights in `weights/` are released under a different license (CC BY 4.0), see `weights/LICENSE.md`.

## How the code was adapted

The original code is written in Python 2 with Theano and Lasagne. We rewrote it for this course in Python 3 with PyTorch and Partitura. It is not a copy of the files: the structure, the algorithms and the training procedure of the original are kept, but the implementation is new.

| File of this folder | Based on (original repository) | What changed |
|:---|:---|:---|
| `melody_cnn.py` | `nn_models/cnn.py`, `build_CNN_model` in `melody_extractor/trainer.py` | The network (two convolutional layers and the decoder with tied weights, a sigmoid, and the masking with the input) is re-implemented in PyTorch, without the Theano/Lasagne layers. Kernels are flipped when loading the original weights, since Theano computes convolutions and PyTorch computes cross-correlations. The saliency maps and the recurrent model are not included. |
| `melody_utils.py` | `data_handling/parse_data.py`, `utils/pianoroll_utils.py`, `melody_extractor/misc_tools.py`, `melody_extractor/graph_tools.py` | Ported to Python 3. The piano rolls are computed with Partitura (`compute_pianoroll`) instead of the original function (durations are rounded instead of offsets, so a few pixels can differ). The threshold is computed exactly, as in the paper, by clustering the note probabilities (the original code clusters the pixels). The graph search of the paper (Algorithm 1) is re-implemented as a single pass over the notes, which is equivalent to Bellman-Ford for this graph. Music21 is not needed. |
| `melody_training.py` | `melody_extractor/trainer.py`, `melody_extractor/settings.py` | Same train/validation/test split, loss, optimizer (AdaDelta, learning rate 1), initialization (Glorot uniform) and early stopping. Differences: the melody is transposed by the same interval in a whole window (the original code moved each pixel independently), the mini-batches are shuffled in every epoch, and the "target switching" trick, the L1 regularization and the dropout (which have no effect in the original code) are not included. |
| `skyline.py` | `melody_extractor/skyline.py` | The skyline algorithm for lists of notes, re-implemented with numpy. The animation is new. |
| `graph_animation.py`, `convert_theano_weights.py` | — | New for this course. |

Any error in the adaptation is ours and not of the authors of the original code.
