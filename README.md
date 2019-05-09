# QNN
Implementation of a quantum neural network using [Strawberry Fields](https://github.com/xanaduai/strawberryfields) and
TensorFlow.

Aimed to be a proof-of-concept for a realistic QNN, based on the model proposed in the paper
[arXiv:1806.06971](https://arxiv.org/pdf/1806.06871.pdf).

## Project Structure
The `curve_fit` folder contains an implemetation of a QNN to perform curve fitting to noisy data. The way this is performed
is also described in the [paper](https://arxiv.org/pdf/1806.06871.pdf) mentioned above.

There are two different implementations of the QNN. One is exactly as described in the paper. The other has an ancilla mode,
which is entangled with the main mode using a beam splitter, then a post-selection measurement is performed. This overall
results in a non-Gaussian operation on the main mode, which is used to replace the Kerr gate.

- `curve_fit/run_curve_fit_x.py` implements the QNN described in the paper (with a Kerr gate)
- `curve_fit/run_curve_fit.py` uses an ancilla mode with a Fock state

The `curve_fit_cats` folder is the same as above, but using a cat state instead of a Fock state for the ancilla.

The `state_classifier` is a (failed) attempt at using a QNN to classify whether or not the input state is Gaussian.

Within each folder:
- `training/*.npz` - Training data in numpy format
- `params/` - Values of Tensorflow parameters that are loaded prior to training the QNN
- `results/` - Results after training the QNN. Subdirectory names are formatted as `{training set}-{n_in}-{n_post}/` for the
Fock state ancilla, and `{training set}-{n_post}/` for the cat state ancilla. Within these directories:
  - The `checkpoint` and `model.*` files are the final parameter values, stored in Tensorflow format
  - `output.npz` contains a set of values that have been input to the QNN, and their corresponding outputs, as well as
  an array containing the value of the loss function at each training epoch
  - `hyperparams.txt` lists the hyperparameters the QNN was trained with
