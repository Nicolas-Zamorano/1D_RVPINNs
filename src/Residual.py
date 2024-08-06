import torch
from Quadrature_Rule import Quadrature_Rule

class Residual:
    """
    Initialize the Residual class.

    Parameters:
    - model_evaluation (torch.nn.Module): Function to evaluate the neural network model
    - quadrature_rule (Quadrature_rule): Quadrature rule for numerical integration
    - gram_elemental_inv_matrix (torch.Tensor): Inverse Gram matrix for subintervals
    - gram_boundary_inv_matrix (torch.Tensor): Inverse Gram matrix for boundary conditions
    - governing_equations (function): Function that evalutes governing equations. it has to have 4 parameters: NN_evaluation, NN_initial_values,  jac_evaluation and initial_values. it has to return 3 torch.Tensor: derivates, governing_equations_evaluation, constrain_vector
    - initial_points (torch.Tensor): Initial points of governing equations
    - initial_values (torch.Tensor): Initial values of governing equations
    - equations_parameters (torch.Tensor): parameters of governing equations
    - constrain_parameter (float): value of contrain constant (default is 0.5)
    """   
    def __init__(self, 
                 model_evaluation: torch.nn.Module,
                 quadrature_rule: Quadrature_Rule,              
                 gram_elemental_inv_matrix: torch.Tensor,    
                 gram_boundary_inv_matrix: torch.Tensor,
                 governing_equations,
                 initial_points: torch.Tensor,               
                 initial_values: torch.Tensor,
                 constrain_parameter: float = 0.5):
            
        self.model_evaluation = model_evaluation
        self.quadrature_rule = quadrature_rule
        self.governing_equations = governing_equations 
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.constrain_parameter = constrain_parameter
        self.update_gram_matrix(gram_elemental_inv_matrix, 
                                gram_boundary_inv_matrix)
            
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
        
            self.gram_elemental_inv_matrix = gram_elemental_inv_matrix
            self.gram_boundary_inv_matrix = gram_boundary_inv_matrix
        
    def residual_value(self):
        """
        Compute the residual value.
        
        Returns:
        - loss_value (torch.Tensor): The computed loss value.
        """
        print("Computing Residual Value...")
        
        NN_evaluation, NN_initial_values, jac_evaluation = self.model_evaluation(self.quadrature_rule.mapped_integration_nodes_single_dimension, 
                                                                                 self.initial_points)
        
        dx, dy, f_1, f_2, constrain_vector = self.governing_equations(NN_evaluation,
                                                                      NN_initial_values,
                                                                      jac_evaluation,
                                                                      self.initial_values)
        
        residual_x_vector = self.quadrature_rule.integrate(f_1) - self.quadrature_rule.integrate(dx)
        residual_y_vector = self.quadrature_rule.integrate(f_2) - self.quadrature_rule.integrate(dy)
        
        residual_vector = torch.concat([residual_x_vector,
                                        residual_y_vector], dim=0)
                
        residual_value = torch.sum(torch.matmul(residual_vector, 
                                                self.gram_elemental_inv_matrix) * residual_vector, dim = 1)
        
        # Use this code if gram_elemental_inv_matrix are diferent for each subinterval
        #
        #x_A = torch.zeros(residual_vector.size())
        #residual_value = torch.zeros(residual_vector.size(0))
        #
        # for i in range(residual_vector.size(0)):
        #     x_A[i] = torch.matmul(residual_vector[i, :], self.gram_elemental_inv_matrix)
        #     residual_value[i] = torch.matmul(x_A[i], residual_vector[i].unsqueeze(0).T.detach().clone())
        #     print(f"\rComputing Residual Value: Processing {i + 1} of {residual_vector.size(0)}", end='', flush=True)
        
        constrain_value = self.constrain_parameter * torch.sum(torch.matmul(constrain_vector, self.gram_boundary_inv_matrix
                                                               ) * constrain_vector,dim = 0, keepdim = True)
        
        loss = torch.nn.L1Loss(reduction='sum')
        
        del dx, dy, f_1, f_2, constrain_vector
        
        return loss(torch.concat([residual_value, constrain_value], dim = 0),
                    torch.zeros(residual_value.size(0) + constrain_value.size(0),requires_grad = False))
