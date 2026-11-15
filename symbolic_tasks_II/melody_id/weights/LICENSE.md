# License of the pre-trained weights

The files `melody_cnn_mozart.pt` and `melody_cnn_pop.pt` contain the weights of the convolutional neural networks trained by the authors of

> Federico Simonetta, Carlos Cancino-Chacón, Stavros Ntalampiras and Gerhard Widmer (2019). "A Convolutional Approach to Melody Line Identification in Symbolic Scores". In *Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)*, Delft, The Netherlands.

on the Mozart dataset (`melody_cnn_mozart.pt`) and on the Pop dataset (`melody_cnn_pop.pt`).

* **Original files:** `nn_kernels_mozart.pkl` and `nn_kernels_pop.pkl` of the repository <https://github.com/LIMUNIMI/Symbolic-Melody-Identification>, released by the authors under the Creative Commons Attribution 4.0 International license (CC BY 4.0).
* **Changes:** the kernels were converted from the format of Theano/Lasagne (Python 2 pickles) to PyTorch tensors with `convert_theano_weights.py`. Since Theano computes true convolutions and PyTorch computes cross-correlations, the kernels were flipped along the pitch and time axes. The values were not modified in any other way.
* **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt these files for any purpose, as long as you give appropriate credit to the authors (reference above), provide a link to the license, and indicate if changes were made.
