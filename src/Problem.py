import torch
from datetime import datetime
from typing import Callable
from Neural_Network import Neural_Network
from Quadrature_Rule import Quadrature_Rule
from Residual import Residual


class IVP:
    """
    Initialize the Initial Value Problem (IVP) class.
    
    Parameters:
    - input_dimension (int): Input dimension of the neural network.
    - output_dimension (int): Output dimension of the neural network.
    - deep_layers (int): Number of deep layers in the neural network.
    - hidden_layers_dimension (int): Dimension of the hidden layers in the neural network.
    - learning_rate (float): Learning rate of the optimizer.
    - collocation_points (int): Number of collocation points used in the quadrature.
    - governing_equations (Callable): Function that defines the governing equations of the problem.
    - initial_points (torch.Tensor): Initial points for the initial value problem (IVP).
    - initial_values (torch.Tensor): Initial values corresponding to the initial points.
    - epochs (int): Number of epochs to train the neural network.
    - domain (Tuple[float, float]): Interval of the domain over which the IVP is solved.
    """

    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 deep_layers: int, 
                 hidden_layers_dimension: int, 
                 learning_rate: float, 
                 epochs: int,
                 collocation_points: torch.Tensor, 
                 governing_equations: Callable, 
                 initial_points: torch.Tensor, 
                 initial_values: torch.Tensor, 
                 gram_matrix_inv: torch.Tensor,
                 gram_boundary_matrix_inv: torch.Tensor):
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.deep_layers = deep_layers
        self.hidden_layers_dimension = hidden_layers_dimension
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.collocation_points = collocation_points

        self.governing_equations = governing_equations
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.constrain_parameter = 0.01
        self.gram_matrix_inv = gram_matrix_inv
        self.gram_boundary_matrix = gram_boundary_matrix_inv
        
        # Neural network initialization
        print("Initializing Neural Network...")
        self.NN = Neural_Network(input_dimension = self.input_dimension, 
                                 output_dimension = self.output_dimension, 
                                 deep_layers = self.deep_layers, 
                                 hidden_layers_dimension = self.hidden_layers_dimension,
                                 optimizer = "Adam",
                                 learning_rate = self.learning_rate)

        # Quadrature rule initialization
        print("Initializing Quadrature Rule...")
        self.quad = Quadrature_Rule(collocation_points = self.collocation_points)

        # Residual initialization
        print("Initializing Residual...")
        self.res = Residual(model_evaluation = self.NN.model_evaluation,
                            quadrature_rule = self.quad,
                            gram_elemental_inv_matrix = self.gram_matrix_inv,
                            gram_boundary_inv_matrix = self.gram_boundary_matrix,
                            governing_equations = self.governing_equations,
                            initial_points = self.initial_points,
                            initial_values = self.initial_values,
                            constrain_parameter = self.constrain_parameter)
    
    def train(self) -> None:
        """
        Train the neural network using the residual method to minimize the error in solving the IVP.
        Also generates plots for the solution and loss evolution.
        """
        loss_relative_error = []
        H_1_relative_error = []

        print(f"{'='*30} Training {'='*30}")
        for epoch in range(self.epochs):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{self.epochs} {'='*20}")
            
            res_value = self.res.residual_value_IVP()
            
            res_error, H_1_error = self.relative_error(self, res_value)
            
            
            print(f"Loss: {res_value.item():.8f}")
            
            self.NN.optimizer_step(res_value)
            
            loss_relative_error.append(res_value.item())
