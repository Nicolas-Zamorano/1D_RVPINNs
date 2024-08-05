import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Neural_Network(torch.nn.Module):
    """
    Initialize the Neural_Network class.

    Parameters:
    - input_dimension (int): Input dimension of the network.
    - output_dimension (int): Output dimension of the network.
    - deep_layers (int): Number of hidden layers (default is 5).
    - hidden_layers_dimension (int): Dimension of each hidden layer (default is 25).
    - activation_function (torch.nn.Module): Activation function (default is torch.nn.Tanh()).
    - optimizer (str): Optimizer name (default is "Adam").
    - learning_rate (float): Learning rate (default is 0.0005).
    """
    def __init__(self, 
                 input_dimension: int,          
                 output_dimension: int,         
                 deep_layers: int = 5,      
                 hidden_layers_dimension: int = 25,      
                 activation_function: torch.nn.Module = torch.nn.Tanh(),  
                 optimizer: str = "Adam",   
                 learning_rate: float = 0.0005):
        
        super().__init__()  
        
        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        self.layer_in = torch.nn.Linear(input_dimension, hidden_layers_dimension)
        self.layer_out = torch.nn.Linear(hidden_layers_dimension, output_dimension)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_layers_dimension, hidden_layers_dimension) for _ in range(deep_layers)])
        self.activation_function = activation_function
        
        self.evaluate = torch.func.vmap(self.forward)  
        self.jacobian = torch.func.vmap(torch.func.jacrev(self.forward)) 
        
        self.optimizer_name = optimizer  
        self.learning_rate = learning_rate  
        
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)  

    def forward(self, 
                Input: torch.Tensor):
        """
        Define how layers are connected.
        
        Parameters:
        - Input (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        Input = Input
        output = self.activation_function(self.layer_in(Input))
        
        for layer in self.middle_layers:
            output = self.activation_function(layer(output))
        
        return self.layer_out(output)

    def model_evaluation(self, 
                         evaluation_points: torch.Tensor, 
                         initial_points: torch.Tensor):
        """
        Evaluate the model and compute the Jacobian.

        Parameters:
        - evaluation_points (torch.Tensor): Nodes for model evaluation.
        - initial_points (torch.Tensor): Initial points for model evaluation.

        Returns:
        - NN_evaluation (torch.Tensor): Model evaluation at the given nodes.
        - NN_initial_values (torch.Tensor): Initial values of the model.
        - jacobian_values (torch.Tensor): Jacobian evaluation at the given nodes.
        """
        evaluation_points = evaluation_points
        initial_points = initial_points
        NN_evaluation = self.evaluate(evaluation_points.unsqueeze(1))  
        NN_initial_values = torch.diagonal(self.evaluate(initial_points.unsqueeze(1)), dim1=-2, dim2=-1)  
        
        jacobian_values = self.jacobian(evaluation_points.unsqueeze(1)).squeeze(2)  
        
        return NN_evaluation, NN_initial_values, jacobian_values
        
    def optimizer_step(self, 
                       loss_value: torch.Tensor):
        """
        Perform an optimization step.
        
        Parameters:
        - loss_value (torch.Tensor): The loss value to minimize.
        
        Updates:
        - Model parameters using the optimizer.
        """
        print("Updating Neural Network...")
        self.optimizer.zero_grad() 
        loss_value.backward(retain_graph=True)  
        self.optimizer.step()  