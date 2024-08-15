import torch
from Quadrature_Rule import Quadrature_Rule
from Neural_Network import Neural_Network
from typing import Callable

class Residual:
    """
    Initialize the Residual class.

    Parameters:
    - model_evaluation (torch.nn.Module): Function to evaluate the neural network model
    - quadrature_rule (Quadrature_rule): Quadrature rule for numerical integration
    - gram_elemental_inv_matrix (torch.Tensor): Inverse Gram matrix for subintervals
    - gram_boundary_inv_matrix (torch.Tensor): Inverse Gram matrix for boundary conditions
    - governing_equations (function): Function that evalutes governing equations. it has to have 2 parameters: collocation_points, NN_evaluation, parameters. it has to return 1 torch.Tensor: governing_equations_evaluation
    - initial_points (torch.Tensor): Initial points of governing equations
    - initial_values (torch.Tensor): Initial values of governing equations
    - governing_equations_parameters (list): parameters of governing equations (default is None)
    - constrain_parameter (float): value of contrain constant (default is 0.5)
    - compure_relative_error 
    """   
    def __init__(self, 
                 neural_network: Neural_Network,
                 quadrature_rule: Quadrature_Rule,              
                 gram_elemental_inv_matrix: torch.Tensor,    
                 gram_boundary_inv_matrix: torch.Tensor,
                 governing_equations: Callable,
                 initial_points: torch.Tensor,               
                 initial_values: torch.Tensor,
                 governing_equations_parameters: list = None,
                 constrain_parameter: float = 0.5,
                 compute_relative_error: bool = False,
                 error_quad: Quadrature_Rule = None,
                 exact_evaluation: Callable = None,
                 exact_jacobian_evaluation: Callable = None, 
                 H_1_exact_norm: torch.Tensor = None):
            
        self.neural_network = neural_network
        self.quadrature_rule = quadrature_rule
        self.governing_equations = governing_equations 
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.governing_equations_parameters = governing_equations_parameters
        self.constrain_parameter = constrain_parameter
        self.update_gram_matrix(gram_elemental_inv_matrix, 
                                gram_boundary_inv_matrix)
        self.compute_relative_error = compute_relative_error
        
        if (self.compute_relative_error == True):
            self.error_quad = error_quad
            self.exact_evaluation = exact_evaluation
            self.exact_jacobian_evaluation = exact_jacobian_evaluation
            self.H_1_exact_norm = H_1_exact_norm
            
            
    def update_gram_matrix(self, 
                           gram_elemental_inv_matrix: torch.Tensor, 
                           gram_boundary_inv_matrix: torch.Tensor):
        """
        Update the Gram matrices.

        Parameters:
        - gram_elemental_inv_matrix (torch.Tensor): Inverse Gram matrix for each polynomial
        - gram_boundary_inv_matrix (torch.Tensor): Inverse Gram matrix for boundary conditions
        """
        print("Updating Gram Matrix...")
        
        with torch.no_grad():
        
            self.gram_elemental_inv_matrix = (1/max(self.quadrature_rule.elements_diameter)) * gram_elemental_inv_matrix
            self.gram_boundary_inv_matrix = gram_boundary_inv_matrix
        
    def residual_value_IVP(self, 
                           compute_error: bool = False):
        """
        Compute the residual value for a Initial Value Problem (IVP).
        
        Returns:
        - loss_value (torch.Tensor): The computed loss value.
        - compute_error (bool): compute_error
        """
        print("Computing Residual Value...")
        
        NN_evaluation = self.quadrature_rule.interpolate(self.neural_network.evaluate)
        NN_jacobian_evaluation = self.quadrature_rule.interpolate(self.neural_network.jacobian)
        NN_initial_value_evalution = self.quadrature_rule.interpolate_boundary(self.neural_network.evaluate)
        
        governing_equations_evaluation = self.quadrature_rule.interpolate(lambda x: self.governing_equations(x,
                                                                                                  NN_evaluation,
                                                                                                  self.governing_equations_parameters))
    
        dx, dy = torch.split(NN_jacobian_evaluation, 1, dim = 1)
        f_1, f_2 = torch.split(governing_equations_evaluation, 1, dim = 1)
        
        constrain_vector = NN_initial_value_evalution - self.initial_values
        
        residual_x_vector = self.quadrature_rule.integrate(f_1) - self.quadrature_rule.integrate(dx)
        residual_y_vector = self.quadrature_rule.integrate(f_2) - self.quadrature_rule.integrate(dy)
        
        residual_vector = torch.concat([residual_x_vector,
                                        residual_y_vector], dim=0)
                        
        residual_value = torch.sum(residual_vector * torch.matmul(residual_vector, 
                                                                  self.gram_elemental_inv_matrix), 
                                   dim = 1, 
                                   keepdim = True)

        
        # Use this code if gram_elemental_inv_matrix are diferent for each subinterval
        #
        #x_A = torch.zeros(residual_vector.size())
        #residual_value = torch.zeros(residual_vector.size(0))
        #
        # for i in range(residual_vector.size(0)):
        #     x_A[i] = torch.matmul(residual_vector[i, :], self.gram_elemental_inv_matrix)
        #     residual_value[i] = torch.matmul(x_A[i], residual_vector[i].unsqueeze(0).T.detach().clone())
        #     print(f"\rComputing Residual Value: Processing {i + 1} of {residual_vector.size(0)}", end='', flush=True)
        
        constrain_value = self.constrain_parameter * torch.matmul(torch.sum(constrain_vector * self.gram_boundary_inv_matrix,
                                                                            dim = 0, 
                                                                            keepdim = True),
                                                                  constrain_vector)
        
        loss = torch.nn.L1Loss(reduction='sum')
        
        res_value = loss(torch.concat([residual_value, constrain_value], dim = 0),
                    torch.zeros(residual_value.size(0) + constrain_value.size(0),requires_grad = False))
        
        if(compute_error == True):
        
            H_1_error = self.H_1_norm()
            
            return res_value, H_1_error
        
        else:
        
            return res_value
