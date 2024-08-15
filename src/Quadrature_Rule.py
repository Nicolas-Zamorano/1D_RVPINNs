import torch
from numpy.polynomial.legendre import leggauss

class Quadrature_Rule:
    """
    Initialize the Quadrature_Rule class.
    
    Parameters:
    - collocation_points (torch.Tensor): Collocation points for integration.
    - quadrature_rule (str): The name of the quadrature rule to use.
    - number_integration_nodes (int): Number of integration nodes.
    - polynomial_degree (int): Degree of the polynomial space.
    """
    def __init__(self,
                 collocation_points: torch.Tensor,
                 initial_points: torch.Tensor,
                 quadrature_rule: str = "Gauss-Legendre", 
                 number_integration_nodes: int = 5, 
                 polynomial_degree: int = 1):
        
        self.initial_points = initial_points
        self.quadrature_rule_name = quadrature_rule
        self.number_integration_nodes = number_integration_nodes
        self.polynomial_degree = polynomial_degree
        
        if self.quadrature_rule_name == "Trapezoidal":
            self.integration_nodes = collocation_points
            
        
        if self.quadrature_rule_name == "Gauss-Legendre":
            integration_nodes, integration_weights = leggauss(number_integration_nodes)
            self.integration_nodes = torch.tensor(integration_nodes, requires_grad = False).unsqueeze(1)
            self.integration_weights = torch.tensor(integration_weights,  requires_grad = False).unsqueeze(1)
        
        
        self.update_collocation_points(collocation_points)
        
    def update_collocation_points(self, 
                                  collocation_points: torch.Tensor):
        """
        Update collocation points and related attributes, such as mapped nodes 
        and weights for integration.
        
        Parameters:
        - collocation_points (torch.Tensor): The new collocation points.
        """
        print("Updating Integration Points...")
        with torch.no_grad():
            self.collocation_points = collocation_points
            
            self.elements_diameter = collocation_points[1:] - collocation_points[:-1]
            self.sum_collocation_points = collocation_points[1:] + collocation_points[:-1]
            self.number_subintervals = self.elements_diameter.size(0)
            
            self.mapped_weights = 0.5 * self.elements_diameter * self.integration_weights.T
            self.mapped_integration_nodes = 0.5 * self.elements_diameter * self.integration_nodes.T + 0.5 * self.sum_collocation_points
            self.mapped_integration_nodes_single_dimension = self.mapped_integration_nodes.view(-1,1)
                        
            self.polynomial_evaluations()
        
    def polynomial_evaluations(self):
        """
        Evaluate polynomials at the integration nodes.
        """
        print("Computing Polynomial Evaluations...")
        with torch.no_grad():
            
            if self.polynomial_degree == 0:
                self.polynomial_evaluation = torch.ones_like(self.mapped_integration_nodes)
            
            if self.polynomial_degree == 1:
                poly_eval_positive = (self.mapped_integration_nodes - self.collocation_points[:-1]) / self.elements_diameter
                poly_eval_negative = (self.collocation_points[1:] - self.mapped_integration_nodes) / self.elements_diameter
                
                self.polynomial_evaluation = torch.stack([poly_eval_positive, poly_eval_negative], dim=0)
      
    
    def interpolate(self,
                    function):
        
        """
        Interpolates function with integration nodes.
        
        Parameters:
        -function (Callable):function to interpolate.
        
        Return:
        -interpolation (torch.Tensor): interpolation of function in integration nodes 
        """
        
        interpolation = function(self.mapped_integration_nodes_single_dimension)
        
        return interpolation
    
    def interpolate_boundary(self,
                             function):
        """
        Interpolates function with boundary nodes.
        
        Parameters:
        -function (Callable): function to interpolte
        
        Return:
        interpolation (torch.Tensor): interpolation of function in boundary nodes. 
        
        """
        
        interpolation = torch.diagonal(function(self.initial_points), dim1=-2, dim2=-1).unsqueeze(1) 

        return interpolation

    def integrate(self, 
                  function_values: torch.Tensor, 
                  multiply_by_test: bool = True):
        """
        Perform integration using the quadrature rule.
        
        Parameters:
        - function_values (torch.Tensor): Function values at the integration nodes.
        - multiply_by_test (bool): Multiply function values by test functions values (default is True).
        Returns:
        - torch.Tensor: The integral values.
        """
        function_values = function_values
        
        if(multiply_by_test == True):
        
            integral_value = torch.zeros((self.number_subintervals, self.polynomial_degree + 1))
            
            for i in range(self.polynomial_degree + 1):
                nodes_value = self.polynomial_evaluation[i, :, :] * function_values.view(self.mapped_integration_nodes.size())
                integral_value[:, i] = torch.sum(self.mapped_weights * nodes_value, dim=1)
        
        else:
            
            nodes_value = function_values.view(self.mapped_integration_nodes.size())
            integral_value = torch.sum(self.mapped_weights * nodes_value, dim=1)
        
        return integral_value
    
    def H_1_norm(self, 
                 function: Callable = None, 
                 jacobian: Callable = None,
                 function_evaluation: torch.Tensor = None,
                 jacobian_evalution: torch.Tensor = None,
                 boundary_evaluation: torch.Tensor = None,
                 ):
        
        if function != None and jacobian != None:
        
            function_evaluation = self.interpolate(function)
            jacobian_evalution = self.interpolate(jacobian)
            boundary_evaluation = self.interpolate_boundary(function)
            
        L_2_norm = self.integrate(function_evaluation**2)
        L_2_jacobian_norm = self.integrate(jacobian_evalution**2)
        boundary_norm = self.integrate(boundary_evaluation**2)
        
        H_1_norm = torch.sum(torch.sqrt(L_2_norm + L_2_jacobian_norm + boundary_norm))
        
        return H_1_norm