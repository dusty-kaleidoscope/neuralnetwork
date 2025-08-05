import node

from typing import List
from operator import add, mul, matmul

import torch

"""TODO
1. Write methods to clear Layer after we are done with it.
2. Fix issues w/ typing, namely for functions and using name of class within itself
"""

"""Conventions

1. Users are responsible for clearing temporary state variables (e.g. the forward_values of InputNodes) after calling the relevant methods (e.g. forward evaluation). Methods are provided for doing so.
2. When using the Layer and ParameterLayer abstractions, users will have to connect the layers to each other.
"""

class Layer:
	"""a basic linear layer with activation

	Users specify the layer by stating the activation function as a string. 

	Supported activations: id, tanh, relu, sigmoid

	Parameters:
	+pre (node.InputNode): InputNode for handling preactivation value
	+activation (node.ActivationNode): activation function
	+post (node.InputNode): InputNode for handling postactivation value

	Methods:
	+user_input (self, torch.Tensor) -> Layer: inserts user input into the forward value of the pre InputNode and returns the Layer
	+clear_state (self) -> Layer: clears state of pre and post then returns the Layer itself
	+forward_pass (self) -> Layer: runs a forward pass through all of the Nodes and returns the Layer itself
	+backward_pass (self) -> Layer: runs a backward pass through all of the Nodes and returns the Layer itself
+disconnect (self) -> Layer: disconnects sources of pre node and targets of post node
"""

	def __init__(self, activation_name: str):
		self.pre = node.InputNode(None, None, None, None)
		
		if (activation_name == "id"):
			self.activation = node.IdNode([self.pre], None)
		elif (activation_name == "tanh"):
			self.activation = node.TanhNode([self.pre], None)
		elif (activation_name == "sigmoid"):
			self.activation = node.SigmoidNode([self.pre], None)
		elif (activation_name == "relu"):
			self.activation = node.ReluNode([self.pre], None)
		else:
			print("Error in creating Layer: Activation function not recognized")
		
		self.pre.targets = [self.activation]
		self.activation.sources = [self.pre]
		self.post = node.InputNode([self.activation], None, None, None)
		self.activation.targets = [self.post]
	
	def user_input(self, X: torch.Tensor) -> "Layer":

		self.pre.forward_value = X

		return self

	def clear_state(self) -> "Layer":

		self.pre.forward_value = None
		self.pre.backward_value = None
		self.post.forward_value = None
		self.post.backward_value = None

		return self

	def forward_pass(self) -> "Layer":

		self.pre.forward_pass()
		self.activation.forward_pass()
		self.post.forward_pass()

		return self

	def backward_pass(self) -> "Layer":

		self.post.backward_pass()
		self.activation.backward_pass()
		self.pre.backward_pass()

		return self
	
	def disconnect(self) -> "Layer":

		self.pre.sources = None
		self.post.sources = None

		return self


class ParameterLayer:
	"""organize a weight, bias, and the operators that help them interact w/ linear layers

	Users initialize and define a ParameterLayer object by specifying a weight Tensor and bias Tensor. Users are responsible for ensuring that their dimensions align.

	Parameters:
	+weight (node.ParameterNode): weight Tensor
	+bias (torch.ParameterNode); bias Tensor. If weight is m-by-n, then bias should be  m-by-1. User is responsible for this.
	+multipler (node.MatMulNode): multiplies input by weight
	+product (node.InputNode): store the intermediate product of the input by the weight
	+adder (node.AddNode): adds bias

	Methods:
	+zero_grad (self) -> ParameterLayer: clears gradients of weight and bias then returns the ParameterLayer itself
	+randomize (self) -> ParameterLayer: 
	+disconnect (self) -> ParameterLayer: disconnects any user-input reliant InputNodes from the multiplier and adder attributes. Returns the ParameterLayer itself
	+clear_state (self) -> ParameterLayer: clears the state from the product
	+forward_pass (self) -> ParameterLayer: runs forward_pass() on the multiplier, then the product, then adder attributes. Returns the ParameterLayer itself
	+backward_pass (self) -> ParameterLayer: runs backward_pass() on the adder, then the product, then the multiplier attributes. Returns the ParameterLayer itself
	+update (self) -> ParameterLayer: updates the parameters using gradients
	+print_params (self) -> None: Prints the parameters
	+to_device(self, torch.device) -> Node: moves weights and biases to the device
	"""

	def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
		
		self.weight = node.ParameterNode(sources = None, targets = None, forward_value = weight, backward_value = torch.zeros(weight.shape)) # needs to be connected to multiplier after the latter is created
		self.bias = node.ParameterNode(sources = None, targets = None, forward_value = bias, backward_value = torch.zeros(bias.shape)) # needs to be connected to adder after the latter is created

		
		self.multiplier = node.MatMulNode(sources = [self.weight], targets = None) # needs to be attached to product after the latter is created
		self.weight.targets = [self.multiplier]
		self.product = node.InputNode(sources = [self.multiplier], targets = None, forward_value = None, backward_value = None) # needs to be attached to adder after the latter is created
		self.multiplier.targets = [self.product]
		self.adder = node.AddNode(sources = [self.product, self.bias], targets = None)
		self.bias.targets = [self.adder]
		self.product.targets = [self.adder]

	def zero_grad(self) -> "ParameterLayer":

		self.weight.zero_grad()
		self.bias.zero_grad()

		return self

	def randomize(self) -> "ParameterLayer":
		
		_, n = self.weight.forward_value.shape
		std = (0.2 / n) ** 0.5
		self.weight.forward_value.normal_(mean = 0, std = std)
		self.bias.forward_value.zero_()

	def disconnect(self) -> "ParameterLayer":
		"""the multiplier may have user input as a source and the adder may have user input as a target
	
		this method will disconnect those"""

		self.multiplier.sources = [self.weight]
		self.adder.targets = None

		return self

	def clear_state(self) -> "ParameterLayer":
		"""the product attribute stores the state from passing user inputs through the ParameterLayer

		this method clears the state"""

		self.product.forward_value = None
		self.product.backward_value = None
		
		return self

	def forward_pass(self) -> "ParameterLayer":
	
		self.multiplier.forward_pass()
		self.product.forward_pass()
		self.adder.forward_pass()

		return self

	def backward_pass(self) -> "ParameterLayer":

		self.adder.backward_pass()
		self.product.backward_pass()
		self.multiplier.backward_pass()

		return self
	
	def update(self, learning_rate: float) -> "ParameterLayer":

		self.weight.forward_value -= learning_rate * self.weight.backward_value
		self.bias.forward_value -= learning_rate * self.bias.backward_value

		return self

	def print_params(self) -> None:

		print(f"The weights are {self.weight.forward_value}.")
		print(f"The biases are {self.bias.forward_value}.")

	def to_device(self, device: torch.device) -> None:

		self.weight.to_device(device)
		self.bias.to_device(device)

		return self
