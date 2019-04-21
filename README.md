# QNN
Implementation of a quantum neural network using [Strawberry Fields](https://github.com/xanaduai/strawberryfields) and
TensorFlow.

Aimed to be a proof-of-concept for a realistic QNN, based on the model proposed in [this](https://arxiv.org/pdf/1806.06871.pdf)
paper.

The `curve_fit` folder contains an implemetation of a QNN to perform curve fitting to noisy data. The way this is performed
is also described in the [paper](https://arxiv.org/pdf/1806.06871.pdf) mentioned above. The architecture of the QNN has been
modified from the one proposed in the paper in that it uses an ancilla mode containing a Fock state instead of a Kerr gate in
order to perform the non-Gaussian operation.

The `curve_fit_cats` folder is the same as above, but using a cat state instead of a Fock state for the ancilla.

The `state_classifier` is a (failed) attempt at using a QNN to classify whether or not the input state is Gaussian.
