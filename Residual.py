import torch

class Residual:
    """
    Initialize the Residual class.

    Parameters:
    - model_evaluation (torch.nn.Module): Function to evaluate the neural network model
    - quadrature_rule (Quadrature_rule): Quadrature rule for numerical integration
    - gram_elemental_inv_matrix (torch.Tensor): Inverse Gram matrix for subintervals
    - gram_boundary_inv_matrix (torch.Tensor): Inverse Gram matrix for boundary conditions
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
                 initial_points: torch.Tensor,               
                 initial_values: torch.Tensor,
                 equations_parameters: torch.Tensor,
                 constrain_parameter: float = 0.5):
            
        self.model_evaluation = model_evaluation
        self.quadrature_rule = quadrature_rule
        self.initial_points = initial_points
        self.initial_values = initial_values
        self.equations_parameters = equations_parameters
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
        
    def governing_equations(self):
        """
        Compute the governing equations.
        
        Returns:
        - dx (torch.Tensor): Derivative of the neural network output with respect to x.
        - dy (torch.Tensor): Derivative of the neural network output with respect to y.
        - f_1 (torch.Tensor): Right side values of the first governing equation.
        - f_2 (torch.Tensor): Right side values of the second governing equation.
        - constrain (torch.Tensor): Constraint values representing the difference between
          the neural network's initial values and the given initial values.
        """
        NN_evaluation, NN_initial_values, jac_evaluation = self.model_evaluation(
            self.quadrature_rule.mapped_integration_nodes_single_dimension, 
            self.initial_points)
        
        mu_max, K, D, s_in= self.equations_parameters
        
        x, s = torch.split(NN_evaluation, 1, dim=1)
        
        dx, dy = torch.split(jac_evaluation, 1, dim=1)
        
        # Computes right side value of governing equations.
        f_1 = (mu_max * s * x) / (s + K) - D * x
        f_2 = (s_in - s) * D - (mu_max * s * x) / (s + K)
        
        constrain = NN_initial_values - self.initial_values
        
        del NN_evaluation, NN_initial_values, jac_evaluation
                
        return dx, dy, f_1, f_2, constrain
    
    def residual_value(self):
        """
        Compute the residual value.
        
        Returns:
        - loss_value (torch.Tensor): The computed loss value.
        """
        print("Computing Residual Value...")
        
        dx, dy, f_1, f_2, constrain_vector = self.governing_equations()
        
        residual_x_vector = self.quadrature_rule.integrate(f_1) - self.quadrature_rule.integrate(dx)
        residual_y_vector = self.quadrature_rule.integrate(f_2) - self.quadrature_rule.integrate(dy)
        
        residual_vector = torch.concat([residual_x_vector,
                                        residual_y_vector], dim=0)
                
        x_A = torch.matmul(residual_vector, self.gram_elemental_inv_matrix)
        residual_value = torch.sum(x_A * residual_vector, dim = 1)
        
        # Use this code if gram_elemental_inv_matrix are diferent for each subinterval
        #
        #x_A = torch.zeros(residual_vector.size())
        #residual_value = torch.zeros(residual_vector.size(0))
        #
        # for i in range(residual_vector.size(0)):
        #     x_A[i] = torch.matmul(residual_vector[i, :], self.gram_elemental_inv_matrix)
        #     residual_value[i] = torch.matmul(x_A[i], residual_vector[i].unsqueeze(0).T.detach().clone())
        #     print(f"\rComputing Residual Value: Processing {i + 1} of {residual_vector.size(0)}", end='', flush=True)
        
        constrain_value = self.constrain_parameter * torch.matmul(constrain_vector, torch.matmul(self.gram_boundary_inv_matrix,
                                                               constrain_vector, dim = 1))
        
        loss = torch.nn.L1Loss(reduction='sum')
        
        del dx, dy, f_1, f_2, constrain_vector, x_A
        
        return loss(torch.concat([residual_value, constrain_value], dim = 0),
                    torch.zeros(residual_value.size(0) + constrain_value.size(0),requires_grad = False))
