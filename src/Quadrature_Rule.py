import torch
from typing import Callable
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
                 boundary_points: torch.Tensor = None,
                 quadrature_rule: str = "Gauss-Legendre", 
                 number_integration_nodes: int = 5, 
                 polynomial_degree: int = 1):
        
        self.boundary_points = boundary_points
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
            if self.quadrature_rule_name == "Gauss-Legendre":
                self.collocation_points = collocation_points
                
                self.elements_diameter = collocation_points[1:] - collocation_points[:-1]
                self.sum_collocation_points = collocation_points[1:] + collocation_points[:-1]
                
                self.mapped_weights = (0.5 * self.elements_diameter * self.integration_weights.T).unsqueeze(-1)
                self.mapped_integration_nodes = 0.5 * self.elements_diameter * self.integration_nodes.T + 0.5 * self.sum_collocation_points
                self.mapped_integration_nodes_single_dimension = self.mapped_integration_nodes.view(-1,1)
                

                
            if self.quadrature_rule_name == "Trapezoid":
                self.collocation_points = collocation_points
                
                self.elements_diameter = collocation_points[1:] - collocation_points[:-1]
            
                self.mapped_integration_nodes = self.integration_nodes
                self.mapped_weights = 0.5 * self.elements_diameter.unsqueeze_(-1)
                self.mapped_integration_nodes_single_dimension = self.mapped_integration_nodes.view(-1,1)
            
            self.nb_subintervals = self.elements_diameter.size(0)
            self.nb_weights = self.mapped_weights.size(1)
            self.update_polynomial_evaluations()
        
    def update_polynomial_evaluations(self):
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
                
                self.polynomial_evaluation = torch.stack([poly_eval_positive, poly_eval_negative], dim = 2).unsqueeze(2)
      
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
        interpolation = torch.diagonal(function(self.boundary_points), dim1=-2, dim2=-1).unsqueeze(1) 

        return interpolation

    def integrate(self, 
                  function: Callable or torch.Tensor,
                  multiply_by_test: bool = False):
        """
        Perform integration using the quadrature rule.
        
        Parameters:
        - function (Callable or torch.Tensor): Function to integrate or function values at the integration nodes.
        - multiply_by_test (bool): Multiply function values by test functions values (default is False).
        
        Returns:
        - torch.Tensor: The integral values in each subinterval of domain.
        """
        if torch.is_tensor(function):
            function_values = function
            
        else:
            function_values = self.interpolate(function)
                
        function_values = function_values.view(self.nb_subintervals, 
                                               self.nb_weights,
                                               function_values.size(1))    
        
        if multiply_by_test == True:
            nodes_value = self.polynomial_evaluation * function_values.unsqueeze(-1)

        else:
            nodes_value = function_values.unsqueeze(-1)
        
        integral_value = torch.sum(self.mapped_weights.unsqueeze(-1) * nodes_value, dim = 1)
        
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
            
        L_2_norm = torch.sum(self.integrate(function_evaluation**2))
        L_2_jacobian_norm = torch.sum(self.integrate(jacobian_evalution**2))
        boundary_norm = torch.sum(boundary_evaluation**2)
        
        H_1_norm = torch.sqrt(L_2_norm + L_2_jacobian_norm + boundary_norm)
        
        return H_1_norm