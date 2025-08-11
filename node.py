from typing import List, Callable
from operator import add, mul, matmul

import torch

"""TODO
1. operators should be allowed to have more than one forward output
2. write methods to clear the node after we are done with it
3. write attach() method to attach two Nodes to each other, replace applicable loc w/ this method
4. Fix issues w/ typing, namely typing for functions and using Node types within the Node class
5. Resolve temporary fix from add_jac
6. Delete and/or rewrite ErrorNode class
"""

"""Conventions:
1. Data is presented as a column vector, and matrix multiplication is on the left.
2. A dataset X consisting of n datapoints with m features each will be written as an m-by-n matrix
3. The derivative of relu at 0 is filled-in to be 0.5.
4. VariableNodes store values and pass them to OperatorNodes. OperatorNodes produce values and pass them to VariableNodes.
"""

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:

	return ((y_true - y_pred) ** 2).mean()

def mse_grad(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
	
	return -2 * (y_true - y_pred) * (1 / y_true.numel())

def bce(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
	"""Binary Cross Entropy

	Assumes data is 1 or 0 (for one-hot encoded data, use cce below)
	"""

	k, n = y_true.shape
	log_predicted_proba = y_pred.log()
	log_predicted_oneminus = (1 - y_pred).log()
	bce = -1 * (y_true * log_predicted_proba + (1 - y_true) * log_predicted_oneminus).sum(dim = 0, keepdim = True) # This should be a 1-by-1 tensor
	return bce / n

def bce_grad(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
	"""assumes none of the predictions are 0 or 1 exactly, i.e. no perfect certainty
	"""
	
	_, n = y_true.shape

	y_pred_recip = y_pred.reciprocal()
	y_pred_oneminus_recip = (1 - y_pred).reciprocal()
	bce_grad = -1 * (y_true * y_pred_recip - (1 - y_true) * (y_pred_oneminus_recip))

	return bce_grad / n


def cce(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
	"""Categorical cross entropy

	Assumes that data is one-hot encoded -- if there are k possible classes and n datapoints, y will be k by n

	Assumes that y_pred is a probability that is never exactly 0, as would be the case if we used a standard softmax head 
	"""

	k, n = y_true.shape
	log_predicted_proba = y_pred.log()
	cce = -1 * (y_true * log_predicted_proba).sum(dim = 0, keepdim = True) # this should be a 1-by-n Tensor containing the cce of each prediction
	return cce / n 
	

def cce_grad(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
	"""Same assumptions as cce
	"""
	k, n = y_true.shape
	recip_predicted_proba = y_pred.reciprocal()
	cce_grad = -1 * (y_true * recip_predicted_proba)

	return cce_grad / n

def id_vect(a: torch.Tensor) -> torch.Tensor:
	"""a is the input of the IdNode
	"""

	return a

def id_jac(x: torch.Tensor, a: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the IdNode
	a is the input of the source of the IdNode
	"""

	return [x]

def tanh_vect(a: torch.Tensor) -> torch.Tensor:

	return a.tanh()

def tanh_jac(x: torch.Tensor, a: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the TanhNode
	a is the input of the source of the TanhNode
	"""

	tanh_prime = 1 - a.tanh()**2
	return [x * tanh_prime]

def sigmoid_vect(a: torch.Tensor) -> torch.Tensor:

	return a.sigmoid()

def sigmoid_jac(x: torch.Tensor, a: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the SigmoidNode
	a is the input of the source of the SigmoidNode
	"""

	sigmoid_a = a.sigmoid()
	sigmoid_prime = (sigmoid_a) * (1 - sigmoid_a)

	return [x * sigmoid_prime]

def relu_vect(a: torch.Tensor) -> torch.Tensor:

	return a.relu()

def relu_jac(x: torch.Tensor, a: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the ReluNode
	a is the input of the source of the ReluNode
	We fill-in relu'(0) to be 0
	"""
	#relu_prime = torch.heaviside(a.cpu() , torch.tensor(0.5)).to(a.device)
	relu_prime = (a > 0).float()

	return [x * relu_prime]


def add_jac(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the AddNode
	a and b are the inputs of the AddNode (must have the same dimension)
	Outputs the gradient propagated to source[0] and source[1] as a list in that order
	We temporarily assume that b is the bias 
	"""

	return [x.clone(), torch.sum(x, dim = 1, keepdim = True)]


def mul_jac(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the MulNode
	a and b are the inputs of the MulNode (must be scalars)
	Outputs the gradients propagated to source[0] and source[1] as a list in that order
	"""

	return [x * b, x * a]

def matmul_jac(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> List[torch.Tensor]:
	"""x is the gradient of the target of the MatMulNode
	a and b are inputs of the MatMulNode (must have compatible dimensions)
	Outputs the gradients propagated to source[0] and source[1] as a list in that order
	"""

	return [x @ b.T, a.T @ x]

class Node:
	"""A node in a neural network

	Note: Our nodes are tensor-valued. As a consequence, when we use our nodes to construct layers, each ``neuron`` may correspond to multiple of what are colloquially called neurons.

	Parameters:
	+sources (List[Node]): The Nodes that point to our Node.
	+targets (List[Node]): The Nodes that our Node points to.

	Methods:
	+foward_pass(self) -> Node: Makes the relevant changes to this Node and/or its targets on a foward pass. Returns the Node itself.
	+backward_pass(self) -> Node: Makes the relevant changes to this Node and/or its targets on a backward pass. Returns the Node itself.
	"""

	def __init__(self, sources: List["Node"], targets: List["Node"]):
		
		self.sources = sources
		self.targets = targets
		

	def forward_pass(self) -> "Node":
		
		return self

	def backward_pass(self) -> "Node":
		
		return self

class VariableNode(Node):
	"""A variable node (input or parameter) in a neural network. Variable nodes receive inputs from their sources (which are operators) and give outputs to operators (which are their targets).

	Parameters:
	+forward_value (Tensor): Stores the relevant value for a forward pass through the network.
	+backward_value (Tensor): Stores the relevant value for a backward pass through the network.

	Methods:

	None
	"""

	def __init__(self, sources: List[Node], targets: List[Node], forward_value: torch.Tensor, backward_value: torch.Tensor):

		super().__init__(sources, targets)
		self.forward_value = forward_value
		self.backward_value = backward_value

class InputNode(VariableNode):
	"""A VariableNode that takes inputs (from user or previous node)

	Parameters:

	No additional parameters.

	Methods:

	No additional methods. Note that InputNodes don't do anything on forward or backward passes. The trainer controls stepping to the next Node.
	"""

class ParameterNode(VariableNode):
	"""A VariableNode that stores parameters and sends them to operators

	Parameters:

	No additional parameters

	Methods:
	+zero_grad(self) -> Node: Sets backwards_value (gradient) to zero. Returns the Node itself.
	+to_device(self, torch.device) -> Node: moves the parameters to the stated torch.device
	Node that ParameterNodes don't do anything on forward passes. OperatorNodes are responsible from retrieving the values from any relevant ParameterNodes.
	"""

	def zero_grad(self) -> Node:

		self.backward_value.zero_()
		
		return self

	def to_device(self, device: torch.device) -> Node:

		self.backward_value = self.backward_value.to(device)
		self.forward_value = self.forward_value.to(device)

		return self

class ErrorNode(Node):
	
	"""A node representing an error function.

	Parameters:
	+error ((torch.Tensor, torch.Tensor) -> float): The error function
	+grad_error ((torch.Tensor,torch.Tensor) -> torch.Tensor): The gradient of the error function
	+forward_value (torch.Tensor): Stores the error
	+backward_value (torch.Tensor): Stores the gradient of the error
	+target (torch.Tensor): The true target

	Methods:
	+compute_error(self, torch.Tensor) -> float: Compute error
	+compute_grad(self, torch.Tensor) -> float: Compute gradient
	+forward_pass(self) -> Node: Set forward_value equal to error. Returns the Node itself.
	+backward_pass(self) -> Node: Set backward_value equal to gradient. Returns the Node itself. Propagates the gradient to the input Nodes.

	Note: For now, we will not use the error node abstraction for training our neural network and instead propagate the gradient of the error within the training control code.
	"""

	def __init__(self, sources: List[Node], targets: List[Node], error, grad_error, forward_value: torch.Tensor, backward_value: torch.Tensor, target: torch.Tensor):

		super().__init__(sources, targets)
		self.error = error
		self.grad_error = grad_error
		self.forward_value = forward_value
		self.backward_value = backward_value
		self.target = target


	def compute_error(self, y_hat: torch.Tensor) -> float:

		return self.error(self.target, y_hat)

	def compute_grad(self, y_hat: torch.Tensor) -> torch.Tensor:

		return self.grad_error(self.target, y_hat)

	def forward_pass(self) -> Node:

		self.forward_value = self.compute_error(sources[0].forward_value)
		
		return self
	
	def backward_pass(self) -> Node:

		grad = self.compute_grad(sources[0].forward_value)
		self.backward_value = grad
		sources[0].backward_value = grad

		return self

class MSENode(ErrorNode):
	
	def __init__(self, sources: List[Node], targets: List[Node], forward_value: torch.Tensor, backward_value: torch.Tensor, target: torch.Tensor):
		"""error and grad_error are mse and mse_grad, resp.
		"""
		super().__init__(sources, targets, mse, mse_grad, forward_value, backward_value, target)

class CCENode(ErrorNode):
	
	def __init__(self, sources: List[Node], targets: List[Node], forward_value: torch.Tensor, backward_value: torch.Tensor, target: torch.Tensor):
		"""error and grad_error are cce and cce_grad, resp.
		"""
		super().__init__(sources, targets, cce, cce_grad, forward_value, backward_value, target)

class OperatorNode(Node):
	"""class representing an Operator Node

	Parameters:
	+forward_function (torch.Tensor,...,torch.Tensor) -> torch.Tensor: function applied during forward propagation
	+backward_function (torch.Tensor,...,torch.Tensor) -> List[torch.Tensor]: propagation of gradients to sources of the OperatorNode. Note that the user and author are responsible for ensuring forward_function and backward_function agree

	Methods:
	+forward_pass(self) -> Node: applies the forward function to the inputs, which are provided by the sources, and forward propagates them to the targets. Returns the OperatorNode itself
	+backward_pass(self) -> Node: receives the previous gradient from the target then updates it and propagates the sources, which have their backward_values updated. Returns the OperatorNode itself 
"""

	def __init__(self, forward_function: Callable[..., List[torch.Tensor]], backward_function: Callable[..., List[torch.Tensor]], sources: List[Node], targets: List[Node]):
		"""assumes that there is a unique target
		"""
		super().__init__(sources, targets)
		self.forward_function = forward_function
		self.backward_function = backward_function

	def forward_pass(self) -> Node:
		"""assumes a single output value
		TODO: allow for more than one output value
		"""
		inputs = [source.forward_value for source in self.sources]
		self.targets[0].forward_value = self.forward_function(*inputs)
		
		return self

	def backward_pass(self) -> Node:
		"""assumes that arguments of backward_function is ordered in (gradients, inputs)
		"""

		grads = [target.backward_value for target in self.targets]
		inputs = [source.forward_value for source in self.sources]
		new_grads = self.backward_function(*grads, *inputs)
		n = len(new_grads)
		for i in range(n): #propagate the gradients
			
			if self.sources[i].backward_value == None:
				self.sources[i].backward_value = new_grads[i]
			else:
				self.sources[i].backward_value += new_grads[i]

		return self
			
class ActivationNode(OperatorNode):
	"""No new attributes and methods, emphasizing it's an activation
	"""

class IdNode(ActivationNode):
	"""Id activation
	"""

	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(id_vect, id_jac, sources, targets)


class TanhNode(ActivationNode):
	"""Tanh activation
	"""

	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(tanh_vect, tanh_jac, sources, targets)


class SigmoidNode(ActivationNode):
	"""Sigmoid activation
	"""

	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(sigmoid_vect, sigmoid_jac, sources, targets)

class ReluNode(ActivationNode):
	"""Relu activation
	"""

	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(relu_vect, relu_jac, sources, targets)

class AddNode(OperatorNode):
	"""Adds two nodes together
	"""
	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(add, add_jac, sources, targets)

class MulNode(OperatorNode):
	"""Multiplies two scalar inputs
	"""	
	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(mul, mul_jac, sources, targets)

class MatMulNode(OperatorNode):
	"""Matrix multiplication of two inputs
	Uses convention that source[0] is on the left, source[1] is on the right
	"""	
	def __init__(self, sources: List[Node], targets: List[Node]):

		super().__init__(matmul, matmul_jac, sources, targets)
