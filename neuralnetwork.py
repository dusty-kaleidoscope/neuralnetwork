import node
from node import mse, mse_grad, cce, cce_grad
import layer

from typing import List
from operator import add, mul, matmul

import torch

"""Conventions
"""

"""TODO
1. make error message print to debugger properly
2. Fix typing esp. for class within itselfand the architecture in __init__
"""

class NeuralNetwork:
	"""a basic feed-forward neural network

	Users specify the network by inputting a list of int (width) and str (activation) pairs.

	Parameters:

	+params (List[ParameterLayer]): the parameters of the neural network, may be saved separately
	+input_layers (List[Layer]): the layers which store states reliant on user inputs

	Methods:
	+randomize_params(self) -> NeuralNetwork: randomizes the parameters of the neural network. Returns the network itself
	+zero_grad(self) -> NeuralNetwork: zeros out the gradients and returns the network itself 
	+clear_state(self) -> NeuralNetwork: clears out the state of the neural network and returns the network itself
	+predict (self, torch.Tensor) -> torch.Tensor: outputs the prediction given a user input
	+train (self, torch.Tensor, torch.Tensor, str, float) -> NeuralNetwork: updates the parameters of the neural network given training data and supported error function
	+save_params (self) -> List[ParameterLayer]: cleans (disconnects and clears state) the parameter layers and returns them
	+print_network (self) -> None: prints the network
	+to_device(self, torch.device): moves parameters to device. User is responsible for putting inputs on device
	"""
	def __init__(self, architecture):

		self.params = []
		self.input_layers = []

		n = len(architecture)

		for i in range(n):
			w, act = architecture[i]
			self.input_layers.append(layer.Layer(act))
			
			if i > 0: # if not the first layer, add weight layer
				v, _ = architecture[i-1]
				self.params.append(layer.ParameterLayer(torch.zeros(w, v), torch.zeros(w, 1)))

		for i in range(n-1):
			self.input_layers[i].post.targets = [self.params[i].multiplier]
			self.params[i].multiplier.sources.append(self.input_layers[i].post)
			self.params[i].adder.targets = [self.input_layers[i + 1].pre]
			self.input_layers[i + 1].pre.sources = [self.params[i].adder]

	def randomize_params(self) -> "NeuralNetwork":
		
		for layer in self.params:
			layer.randomize()

		return self

	def zero_grad(self) -> "NeuralNetwork":

		for layer in self.params:
			layer.zero_grad()

		return self

	def clear_state(self) -> "NeuralNetwork":

		for layer in self.input_layers:
			layer.clear_state()

		for layer in self.params:
			layer.clear_state()

		return self

	def predict(self, X: torch.Tensor) -> torch.Tensor:

		n = len(self.input_layers)

		self.input_layers[0].pre.forward_value = X

		for i in range(n):
			self.input_layers[i].forward_pass()
			if i < n - 1:
				self.params[i].forward_pass()

		return self.input_layers[n-1].post.forward_value

	def to_device(self, device: torch.device) -> "NeuralNetwork":

		for layer in self.params:
			layer.to_device(device)

		return self

	def train(self, X: torch.Tensor, y: torch.Tensor, error: str, learning_rate: float) -> "NeuralNetwork":

		n = len(self.input_layers)

		self.zero_grad()
		self.clear_state()

		y_pred = self.predict(X) # forward pass now complete
		
		if error == "mse":
			self.input_layers[n - 1].post.backward_value = mse_grad(y, y_pred)
		elif error == "cce":
			self.input_layers[n - 1].post.backward_value = cce_grad(y, y_pred)
		else:
			raise ValueError("Neural network training issue: Error not recognized")  

		for i in range(n):
			self.input_layers[n - 1 - i].backward_pass()
			if i < n - 1:
				self.params[n - 1 - 1 - i].backward_pass()

		for i in range(n-1):
			self.params[i].update(learning_rate)

		return self

	def save_params(self) -> List[layer.ParameterLayer]:

		for layer in self.params:
			layer.zero_grad()
			layer.clear_state()

		return self.params 
	
	def print_network(self) -> None:

		for layer in self.params:
			layer.print_params()
