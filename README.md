# Home-Cooked Neural Network

This is a home-cooked neural network training library starting from linear algebra functions as implemented in PyTorch. A demo is [HERE](demo.ipynb) (links to demo Jupyter notebook in repo).

## Structure and Functions

The library is organized into three modules

* `neuralnetwork`
* `layer`
* `node`

where each module depends on the module below it. 

The library currently supports the following:

* Declaring a feedforward neural network with linear layers and relu, sigmoid, tanh, or no activation.
* Making predictions on tensor-valued inputs.
* Training the network using a basic gradient descent algorithm.
* Moving parameters of network to a supported torch.device (CUDA-programmed GPU or Apple MPS-programmed device)
  
The functionality is demonstrated in several examples in the demo.
