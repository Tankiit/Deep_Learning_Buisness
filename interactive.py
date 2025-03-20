import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL import Image
import torchvision.transforms as transforms
import io
import time
import os
from matplotlib.animation import FuncAnimation
from torchvision.models import resnet18

class SimpleCNN(nn.Module):
    def __init__(self, num_filters=16):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(num_filters * 14 * 14, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class SimpleLinear(nn.Module):
    def __init__(self, input_size=784, hidden_size=64):
        super(SimpleLinear, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten 28x28 to 784
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GradientDescentVisualizer:
    def __init__(self, model_type='linear'):
        # Initialize dummy data (MNIST-like)
        self.create_synthetic_data()
        
        # Model 
        if model_type == 'linear':
            self.model = SimpleLinear()
        else:
            self.model = SimpleCNN()
            
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.gradient_history = []
        self.weight_history = []
        self.step_history = []
        
        # For loss landscape
        self.w1_landscape = None
        self.w2_landscape = None
        self.loss_landscape = None
        
    def create_synthetic_data(self):
        """Create a synthetic dataset to mimic MNIST"""
        # Create 100 samples
        self.x_data = torch.randn(100, 1, 28, 28)
        # Create random labels (0-9)
        self.y_data = torch.randint(0, 10, (100,))
        
        # Split into train/test
        train_size = 80
        self.x_train = self.x_data[:train_size]
        self.y_train = self.y_data[:train_size]
        self.x_test = self.x_data[train_size:]
        self.y_test = self.y_data[train_size:]
    
    def compute_loss_landscape(self, param_index_1=0, param_index_2=1, grid_size=20, alpha=1.0):
        """Compute the loss landscape around current weights"""
        # Make a copy of the current model parameters
        current_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Get the two parameters we want to visualize
        params = list(self.model.parameters())
        param1 = params[param_index_1].clone().detach()
        param2 = params[param_index_2].clone().detach()
        
        # Flatten parameters
        flat_param1 = param1.view(-1)
        flat_param2 = param2.view(-1)
        
        # Choose two random directions in the parameter space
        # We'll use the first two principal components of the parameters
        direction1 = torch.randn_like(flat_param1)
        direction2 = torch.randn_like(flat_param2)
        
        # Normalize the directions
        direction1 = direction1 / direction1.norm()
        direction2 = direction2 / direction2.norm()
        
        # Create a grid of perturbations
        grid = np.linspace(-alpha, alpha, grid_size)
        self.w1_landscape, self.w2_landscape = np.meshgrid(grid, grid)
        self.loss_landscape = np.zeros_like(self.w1_landscape)
        
        # Calculate the loss at each point in the grid
        for i in range(grid_size):
            for j in range(grid_size):
                # Perturb the parameters
                perturbed_param1 = flat_param1 + self.w1_landscape[i, j] * direction1
                perturbed_param2 = flat_param2 + self.w2_landscape[i, j] * direction2
                
                # Update the model with perturbed parameters
                params[param_index_1].data = perturbed_param1.view_as(param1)
                params[param_index_2].data = perturbed_param2.view_as(param2)
                
                # Compute loss
                outputs = self.model(self.x_train)
                loss = self.criterion(outputs, self.y_train)
                self.loss_landscape[i, j] = loss.item()
                
        # Restore original parameters
        for i, param in enumerate(current_params):
            list(self.model.parameters())[i].data = param
            
    def train(self, learning_rate=0.01, momentum=0.0, optimizer_type='sgd', epochs=20,
              compute_landscape=True, record_grad=True, batch_size=16):
        """Train the model and record metrics for visualization"""
        # Reset model
        if isinstance(self.model, SimpleCNN):
            self.model = SimpleCNN()
        else:
            self.model = SimpleLinear()
            
        # Set optimizer
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
            
        # Reset history
        self.loss_history = []
        self.accuracy_history = []
        self.gradient_history = []
        self.weight_history = []
        self.step_history = []
        
        # Compute initial loss landscape if needed
        if compute_landscape:
            self.compute_loss_landscape()
            
        # Track weight trajectory for the first weight parameter
        params = list(self.model.parameters())
        first_param = params[0].clone().detach()
        flat_param = first_param.view(-1)
        direction1 = torch.randn_like(flat_param)
        direction2 = torch.randn_like(flat_param)
        direction1 = direction1 / direction1.norm()
        direction2 = direction2 / direction2.norm()
        
        # Training loop
        for epoch in range(epochs):
            # Training mode
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Mini-batch training
            indices = torch.randperm(len(self.x_train))
            for i in range(0, len(self.x_train), batch_size):
                # Get mini-batch
                idx = indices[i:i+batch_size]
                x_batch = self.x_train[idx]
                y_batch = self.y_train[idx]
                
                # Forward pass
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Record gradients
                if record_grad and epoch % 2 == 0:
                    grad_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    self.gradient_history.append(grad_norm)
                
                self.optimizer.step()
                
                # Record weight trajectory
                if compute_landscape and epoch % 2 == 0:
                    current_flat_param = params[0].clone().detach().view(-1)
                    proj1 = torch.dot(current_flat_param - flat_param, direction1).item()
                    proj2 = torch.dot(current_flat_param - flat_param, direction2).item()
                    self.weight_history.append((proj1, proj2))
                    self.step_history.append(epoch)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
            # Epoch statistics
            epoch_loss = running_loss / (len(self.x_train) / batch_size)
            epoch_acc = 100 * correct / total
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_acc)
                
        # Final loss landscape
        if compute_landscape:
            self.compute_loss_landscape()
            
    def test(self):
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = self.model(self.x_test)
            _, predicted = torch.max(outputs.data, 1)
            total = self.y_test.size(0)
            correct = (predicted == self.y_test).sum().item()
            
        accuracy = 100 * correct / total
        return accuracy
    
    def visualize_gradients(self):
        """Plot gradient norms during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.gradient_history)
        plt.xlabel('Optimization Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Magnitude During Training')
        plt.grid(True)
        plt.show()
        
    def visualize_loss_landscape(self, show_trajectory=True):
        """Visualize the loss landscape and optimization trajectory"""
        fig = plt.figure(figsize=(15, 12))
        
        # 3D surface plot
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(self.w1_landscape, self.w2_landscape, self.loss_landscape, 
                              cmap=cm.coolwarm, alpha=0.8, linewidth=0)
        if show_trajectory and len(self.weight_history) > 0:
            w1_path = [w[0] for w in self.weight_history]
            w2_path = [w[1] for w in self.weight_history]
            # Interpolate loss values along the path
            loss_path = []
            for w1, w2 in zip(w1_path, w2_path):
                # Find nearest grid point
                i = np.argmin(np.abs(self.w1_landscape[0, :] - w1))
                j = np.argmin(np.abs(self.w2_landscape[:, 0] - w2))
                if i < len(self.w1_landscape) and j < len(self.w2_landscape):
                    loss_path.append(self.loss_landscape[j, i])
                else:
                    loss_path.append(np.nan)
                    
            ax1.plot(w1_path, w2_path, loss_path, 'r-', linewidth=2, label='Optimization Path')
            
        ax1.set_xlabel('Weight Direction 1')
        ax1.set_ylabel('Weight Direction 2')
        ax1.set_zlabel('Loss')
        ax1.set_title('3D Loss Landscape')
        
        # Contour plot (top view)
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(self.w1_landscape, self.w2_landscape, self.loss_landscape, 
                             levels=50, cmap=cm.coolwarm)
        fig.colorbar(contour, ax=ax2)
        
        if show_trajectory and len(self.weight_history) > 0:
            # Plot optimization trajectory
            w1_path = [w[0] for w in self.weight_history]
            w2_path = [w[1] for w in self.weight_history]
            ax2.plot(w1_path, w2_path, 'o-', color='black', markersize=3, linewidth=1)
            
            # Add arrows to show direction
            step_size = max(1, len(w1_path) // 10)
            for i in range(0, len(w1_path) - step_size, step_size):
                ax2.annotate('', xy=(w1_path[i+step_size], w2_path[i+step_size]), 
                         xytext=(w1_path[i], w2_path[i]),
                         arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
                
        ax2.set_xlabel('Weight Direction 1')
        ax2.set_ylabel('Weight Direction 2')
        ax2.set_title('Loss Contour with Optimization Path')
        
        # Training curves
        ax3 = fig.add_subplot(223)
        ax3.plot(self.loss_history, 'b-', label='Training Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        ax3.grid(True)
        
        ax4 = fig.add_subplot(224)
        ax4.plot(self.accuracy_history, 'g-', label='Training Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Training Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

    def visualize_optimization_methods(self, learning_rates=[0.1, 0.01, 0.001], 
                                      optimizers=['sgd', 'sgd_momentum', 'adam', 'rmsprop']):
        """Compare different optimization methods"""
        # Store results
        all_losses = {}
        all_accuracies = {}
        
        # Train with different optimizers
        for opt in optimizers:
            all_losses[opt] = {}
            all_accuracies[opt] = {}
            
            for lr in learning_rates:
                momentum = 0.9 if opt == 'sgd_momentum' else 0.0
                opt_type = 'sgd' if opt in ['sgd', 'sgd_momentum'] else opt
                
                self.train(learning_rate=lr, momentum=momentum, optimizer_type=opt_type, 
                         compute_landscape=False, record_grad=False)
                
                all_losses[opt][lr] = self.loss_history.copy()
                all_accuracies[opt][lr] = self.accuracy_history.copy()
        
        # Visualization
        fig, axs = plt.subplots(len(optimizers), 2, figsize=(15, 5*len(optimizers)))
        
        for i, opt in enumerate(optimizers):
            # Loss curves
            for lr in learning_rates:
                axs[i, 0].plot(all_losses[opt][lr], label=f'LR = {lr}')
            axs[i, 0].set_xlabel('Epoch')
            axs[i, 0].set_ylabel('Loss')
            axs[i, 0].set_title(f'Loss Curves - {opt.upper()}')
            axs[i, 0].legend()
            axs[i, 0].grid(True)
            
            # Accuracy curves
            for lr in learning_rates:
                axs[i, 1].plot(all_accuracies[opt][lr], label=f'LR = {lr}')
            axs[i, 1].set_xlabel('Epoch')
            axs[i, 1].set_ylabel('Accuracy (%)')
            axs[i, 1].set_title(f'Accuracy Curves - {opt.upper()}')
            axs[i, 1].legend()
            axs[i, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Final comparison table
        print("\nFinal Results Comparison")
        print("-----------------------")
        print(f"{'Optimizer':<12} {'Learning Rate':<12} {'Final Loss':<12} {'Final Accuracy':<12}")
        print(f"{'-'*50}")
        
        for opt in optimizers:
            for lr in learning_rates:
                final_loss = all_losses[opt][lr][-1]
                final_acc = all_accuracies[opt][lr][-1]
                print(f"{opt:<12} {lr:<12.4f} {final_loss:<12.4f} {final_acc:<12.2f}%")
                
    def process_uploaded_image(self, image_data):
        """Process an uploaded image for use with the model"""
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale and resize to 28x28
        img = img.convert('L').resize((28, 28))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        return img_tensor, img

def create_interactive_dashboard():
    visualizer = GradientDescentVisualizer()
    
    def on_train_button_click(b):
        with out:
            clear_output(wait=True)
            print(f"Training with: LR={lr_slider.value}, Optimizer={optimizer_dropdown.value}")
            print(f"Computing loss landscape: {landscape_checkbox.value}")
            
            # Handle momentum for SGD
            momentum = momentum_slider.value if optimizer_dropdown.value == 'sgd' else 0.0
            
            # Map optimizer dropdown to actual optimizer
            opt_map = {
                'sgd': 'sgd',
                'sgd_momentum': 'sgd',
                'adam': 'adam',
                'rmsprop': 'rmsprop'
            }
            
            # Start timer
            start_time = time.time()
            
            # Train the model
            visualizer.train(
                learning_rate=lr_slider.value,
                momentum=momentum,
                optimizer_type=opt_map[optimizer_dropdown.value],
                epochs=epoch_slider.value,
                compute_landscape=landscape_checkbox.value,
                batch_size=batch_slider.value
            )
            
            # End timer
            elapsed_time = time.time() - start_time
            
            # Test accuracy
            test_acc = visualizer.test()
            
            # Display results
            print(f"\nTraining completed in {elapsed_time:.2f} seconds")
            print(f"Final training loss: {visualizer.loss_history[-1]:.4f}")
            print(f"Final training accuracy: {visualizer.accuracy_history[-1]:.2f}%")
            print(f"Test accuracy: {test_acc:.2f}%")
            
            # Display plots based on what was computed
            if landscape_checkbox.value:
                visualizer.visualize_loss_landscape(show_trajectory=True)
            else:
                # Just show training curves
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(visualizer.loss_history, 'b-')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(visualizer.accuracy_history, 'g-')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Training Accuracy')
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
    
    def on_compare_button_click(b):
        with out:
            clear_output(wait=True)
            print("Comparing optimization methods...")
            visualizer.visualize_optimization_methods(
                learning_rates=[0.1, 0.01, 0.001],
                optimizers=['sgd', 'sgd_momentum', 'adam', 'rmsprop']
            )
    
    def on_upload_change(change):
        if not change.new:
            return
            
        # Read the file content
        content = change.new[0]['content']
        
        with out:
            clear_output(wait=True)
            
            # Process the image
            try:
                img_tensor, img = visualizer.process_uploaded_image(content)
                
                # Display the processed image
                plt.figure(figsize=(5, 5))
                plt.imshow(img, cmap='gray')
                plt.title('Processed Input Image (28x28)')
                plt.axis('off')
                plt.show()
                
                # Use the current model to predict
                visualizer.model.eval()
                with torch.no_grad():
                    output = visualizer.model(img_tensor)
                    _, predicted = torch.max(output, 1)
                    prob = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Show prediction
                print(f"Model prediction: {predicted.item()}")
                
                # Show prediction probabilities
                plt.figure(figsize=(10, 4))
                plt.bar(range(10), prob.numpy())
                plt.xlabel('Digit Class')
                plt.ylabel('Probability')
                plt.title('Prediction Probabilities')
                plt.xticks(range(10))
                plt.grid(axis='y')
                plt.show()
                
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Create widgets
    model_dropdown = widgets.Dropdown(
        options=[('Linear Network', 'linear'), ('Simple CNN', 'cnn')],
        value='linear',
        description='Model:',
    )
    
    optimizer_dropdown = widgets.Dropdown(
        options=[
            ('SGD', 'sgd'),
            ('SGD+Momentum', 'sgd_momentum'),
            ('Adam', 'adam'),
            ('RMSprop', 'rmsprop')
        ],
        value='sgd',
        description='Optimizer:',
    )
    
    lr_slider = widgets.FloatLogSlider(
        value=0.01,
        base=10,
        min=-3,  # 10^-3
        max=0,   # 10^0
        step=0.1,
        description='Learning Rate:',
        continuous_update=False
    )
    
    momentum_slider = widgets.FloatSlider(
        value=0.9,
        min=0.0,
        max=0.99,
        step=0.01,
        description='Momentum:',
        continuous_update=False,
        disabled=False
    )
    
    epoch_slider = widgets.IntSlider(
        value=20,
        min=5,
        max=100,
        step=5,
        description='Epochs:',
        continuous_update=False
    )
    
    batch_slider = widgets.IntSlider(
        value=16,
        min=1,
        max=64,
        step=1,
        description='Batch Size:',
        continuous_update=False
    )
    
    landscape_checkbox = widgets.Checkbox(
        value=True,
        description='Compute Loss Landscape',
        disabled=False
    )
    
    train_button = widgets.Button(
        description='Train Model',
        button_style='success',
        tooltip='Start training with current settings'
    )
    train_button.on_click(on_train_button_click)
    
    compare_button = widgets.Button(
        description='Compare Optimizers',
        button_style='info',
        tooltip='Compare different optimization methods'
    )
    compare_button.on_click(on_compare_button_click)
    
    upload_button = widgets.FileUpload(
        accept='image/*',
        multiple=False,
        description='Upload Image'
    )
    upload_button.observe(on_upload_change, names='value')
    
    # Output area
    out = widgets.Output()
    
    # Update momentum slider visibility based on optimizer selection
    def on_optimizer_change(change):
        if change.new == 'sgd_momentum':
            momentum_slider.disabled = False
        else:
            momentum_slider.disabled = True
    
    optimizer_dropdown.observe(on_optimizer_change, names='value')
    
    # Layout
    ui = widgets.VBox([
        widgets.HBox([model_dropdown, optimizer_dropdown]),
        widgets.HBox([lr_slider, momentum_slider]),
        widgets.HBox([epoch_slider, batch_slider]),
        widgets.HBox([landscape_checkbox]),
        widgets.HBox([train_button, compare_button, upload_button]),
        out
    ])
    
    display(ui)
    
    # Initialize
    with out:
        print("Ready to train. Set parameters and click 'Train Model'")
        print("Note: Computing the loss landscape requires additional time but enables visualization of the optimization path")

# Example function to visualize gradient descent steps
def visualize_2d_gradient_descent(function_type='bowl', learning_rate=0.1, momentum=0.0,
                              steps=20, starting_point=None):
    """
    Visualize gradient descent in 2D for simple functions
    
    Parameters:
    function_type: 'bowl', 'ravine', or 'saddle'
    learning_rate: step size
    momentum: momentum coefficient
    steps: number of optimization steps
    starting_point: (x, y) starting coordinates, or None for random
    """
    # Define functions and their gradients
    if function_type == 'bowl':
        # Simple quadratic bowl: f(x,y) = x^2 + y^2
        def f(x, y):
            return x**2 + y**2
        
        def grad_f(x, y):
            return np.array([2*x, 2*y])
            
        x_range = (-2, 2)
        y_range = (-2, 2)
        title = "Gradient Descent on Quadratic Bowl"
        
    elif function_type == 'ravine':
        # Ravine function: f(x,y) = 10*x^2 + y^2
        def f(x, y):
            return 10*x**2 + y**2
        
        def grad_f(x, y):
            return np.array([20*x, 2*y])
            
        x_range = (-2, 2)
        y_range = (-4, 4)
        title = "Gradient Descent on Ravine Function"
        
    elif function_type == 'saddle':
        # Saddle point: f(x,y) = x^2 - y^2
        def f(x, y):
            return x**2 - y**2
        
        def grad_f(x, y):
            return np.array([2*x, -2*y])
            
        x_range = (-2, 2)
        y_range = (-2, 2)
        title = "Gradient Descent on Saddle Point"
        
    else:
        # Non-convex function with multiple minima
        def f(x, y):
            return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)
        
        def grad_f(x, y):
            dx = np.cos(x) * np.cos(y) + 0.2 * x
            dy = -np.sin(x) * np.sin(y) + 0.2 * y
            return np.array([dx, dy])
            
        x_range = (-4, 4)
        y_range = (-4, 4)
        title = "Gradient Descent on Non-convex Function"
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 6))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    
    # Create the grid
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(xi, yi) for xi in x] for yi in y])
    
    # Plot the surface
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    ax1.set_title('Function Surface')
    
    # Plot the contour
    contour = ax2.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)
    plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Gradient Descent Path')
    
    # Initial point
    if starting_point is None:
        # Random starting point
        if function_type == 'ravine':
            x0 = np.random.uniform(-1, 1)
            y0 = np.random.uniform(-3, 3)
        else:
            x0 = np.random.uniform(x_range[0]*0.7, x_range[1]*0.7)
            y0 = np.random.uniform(y_range[0]*0.7, y_range[1]*0.7)
    else:
        x0, y0 = starting_point
        
    # Perform gradient descent
    points = [(x0, y0)]
    x_curr, y_curr = x0, y0
    velocity = np.array([0.0, 0.0])  # For momentum
    
    for i in range(steps):
        # Compute gradient
        grad = grad_f(x_curr, y_curr)
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        x_curr += velocity[0]
        y_curr += velocity[1]
        
        # Store the point
        points.append((x_curr, y_curr))
    
    # Convert points to arrays for plotting
    points = np.array(points)
    
    # Plot the path in 3D
    ax1.plot(points[:, 0], points[:, 1], [f(p[0], p[1]) for p in points], 
             'r-o', markersize=4, linewidth=2)
    
    # Plot the path in 2D
    ax2.plot(points[:, 0], points[:, 1], 'r-o', markersize=4, linewidth=2)
    
    # Add arrows to show direction in 2D
    arrow_indices = np.linspace(0, len(points)-2, min(10, len(points)-1)).astype(int)
    for i in arrow_indices:
        ax2.annotate('', xy=(points[i+1][0], points[i+1][1]), 
                   xytext=(points[i][0], points[i][1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.suptitle(f"{title}\nLearning Rate: {learning_rate}, Momentum: {momentum}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print final results
    print(f"Starting point: ({points[0][0]:.4f}, {points[0][1]:.4f}), f(x,y) = {f(points[0][0], points[0][1]):.4f}")
    print(f"Final point after {steps} steps: ({points[-1][0]:.4f}, {points[-1][1]:.4f}), f(x,y) = {f(points[-1][0], points[-1][1]):.4f}")
    print(f"Function decrease: {f(points[0][0], points[0][1]) - f(points[-1][0], points[-1][1]):.4f}")

# Interactive visualization for gradient descent on different functions  
def create_interactive_gd_visualizer():
    """Create an interactive widget for visualizing gradient descent"""
    
    def on_run_button_click(b):
        with out:
            clear_output(wait=True)
            visualize_2d_gradient_descent(
                function_type=function_dropdown.value,
                learning_rate=lr_slider.value,
                momentum=momentum_slider.value,
                steps=steps_slider.value
            )
    
    # Create widgets
    function_dropdown = widgets.Dropdown(
        options=[
            ('Quadratic Bowl', 'bowl'),
            ('Ravine (Ill-conditioned)', 'ravine'),
            ('Saddle Point', 'saddle'),
            ('Non-convex', 'nonconvex')
        ],
        value='bowl',
        description='Function:',
    )
    
    lr_slider = widgets.FloatLogSlider(
        value=0.1,
        base=10,
        min=-3,  # 10^-3
        max=0,   # 10^0
        step=0.1,
        description='Learning Rate:',
        continuous_update=False
    )
    
    momentum_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.99,
        step=0.05,
        description='Momentum:',
        continuous_update=False
    )
    
    steps_slider = widgets.IntSlider(
        value=20,
        min=5,
        max=100,
        step=5,
        description='Steps:',
        continuous_update=False
    )
    
    run_button = widgets.Button(
        description='Run Visualization',
        button_style='success',
        tooltip='Visualize gradient descent'
    )
    run_button.on_click(on_run_button_click)
    
    # Output area
    out = widgets.Output()
    
    # Layout
    ui = widgets.VBox([
        widgets.HBox([function_dropdown, lr_slider]),
        widgets.HBox([momentum_slider, steps_slider]),
        run_button,
        out
    ])
    
    display(ui)
    
    # Initial visualization
    with out:
        visualize_2d_gradient_descent()

# Class for visualizing neural network gradients during backpropagation
class BackpropVisualizer:
    def __init__(self):
        # Create a simple image input
        self.create_sample_data()
        
        # Simple model with hook for gradient visualization
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Register hooks to capture gradients
        self.gradients = {}
        self.activations = {}
        self._register_hooks()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def create_sample_data(self):
        """Create a sample digit-like image"""
        # Create a blank 28x28 image
        img = np.zeros((28, 28), dtype=np.float32)
        
        # Draw a simple digit-like pattern (e.g., number 3)
        for i in range(5, 23):
            # Top horizontal line
            img[5, i] = 1.0
            # Middle horizontal line
            img[14, i] = 1.0
            # Bottom horizontal line
            img[22, i] = 1.0
            
        # Right vertical lines
        for i in range(5, 23):
            img[i, 22] = 1.0
            
        # Add some noise
        img += 0.05 * np.random.randn(28, 28)
        img = np.clip(img, 0, 1)
        
        # Convert to tensor
        self.x_data = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        self.y_data = torch.tensor([3])  # Label as 3
        
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
            
        # Register hooks for each layer
        self.model[1].register_forward_hook(get_activation('fc1'))
        self.model[3].register_forward_hook(get_activation('fc2'))
        
    def compute_gradients(self):
        """Forward and backward pass to compute gradients"""
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(self.x_data)
        
        # Save output gradients for visualization
        output.register_hook(get_gradient('output'))
        self.activations['fc1'].register_hook(get_gradient('fc1_act'))
        
        # Backward pass
        loss = self.criterion(output, self.y_data)
        loss.backward()
        
        # Get weight gradients
        self.gradients['fc1_weight'] = self.model[1].weight.grad.detach()
        self.gradients['fc2_weight'] = self.model[3].weight.grad.detach()
        
        return loss.item(), output
        
    def visualize_image_input(self):
        """Visualize the input image"""
        plt.figure(figsize=(6, 6))
        plt.imshow(self.x_data.squeeze().numpy(), cmap='gray')
        plt.title(f'Input Image (Label: {self.y_data.item()})')
        plt.colorbar()
        plt.axis('off')
        plt.show()
        
    def visualize_gradients(self):
        """Visualize gradients flowing through the network"""
        loss, output = self.compute_gradients()
        
        # Compute predictions
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Input image
        ax1 = fig.add_subplot(231)
        ax1.imshow(self.x_data.squeeze().numpy(), cmap='gray')
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # 2. First layer weights (visualize a subset)
        ax2 = fig.add_subplot(232)
        weights_img = self.model[1].weight[:10].view(10, 28, 28).mean(0).numpy()
        im = ax2.imshow(weights_img, cmap='coolwarm')
        ax2.set_title('First Layer Weights\n(Mean across neurons)')
        plt.colorbar(im, ax=ax2)
        ax2.axis('off')
        
        # 3. First layer weight gradients
        ax3 = fig.add_subplot(233)
        weight_grads = self.gradients['fc1_weight'][:10].view(10, 28, 28).mean(0).numpy()
        im = ax3.imshow(weight_grads, cmap='coolwarm')
        ax3.set_title('First Layer Weight Gradients\n(Mean across neurons)')
        plt.colorbar(im, ax=ax3)
        ax3.axis('off')
        
        # 4. Hidden layer activations (first 100 neurons)
        ax4 = fig.add_subplot(234)
        hidden_acts = self.activations['fc1'][0, :100].numpy()
        ax4.bar(range(len(hidden_acts)), hidden_acts)
        ax4.set_title('Hidden Layer Activations\n(First 100 neurons)')
        ax4.set_xlabel('Neuron Index')
        ax4.set_ylabel('Activation')
        
        # 5. Hidden layer gradients (first 100 neurons)
        ax5 = fig.add_subplot(235)
        hidden_grads = self.gradients['fc1_act'][0, :100].numpy()
        ax5.bar(range(len(hidden_grads)), hidden_grads)
        ax5.set_title('Hidden Layer Gradients\n(First 100 neurons)')
        ax5.set_xlabel('Neuron Index')
        ax5.set_ylabel('Gradient')
        
        # 6. Output probabilities and gradients
        ax6 = fig.add_subplot(236)
        output_bars = ax6.bar(range(10), probabilities.numpy())
        ax6.set_title(f'Output Probabilities\nPrediction: {predicted.item()}, True: {self.y_data.item()}, Loss: {loss:.4f}')
        ax6.set_xlabel('Digit Class')
        ax6.set_ylabel('Probability')
        ax6.set_xticks(range(10))
        
        # Add output gradients as text
        output_grads = self.gradients['output'][0].numpy()
        for i, (p, g) in enumerate(zip(probabilities, output_grads)):
            if abs(g) > 0.01:  # Only show significant gradients
                ax6.text(i, p.item() + 0.05, f'∇: {g:.2f}', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: gradient flow magnitude through the network
        plt.figure(figsize=(10, 6))
        gradient_norms = {
            'Input': torch.mean(torch.abs(self.x_data.grad)).item() if self.x_data.grad is not None else 0,
            'Layer 1 Weights': torch.norm(self.gradients['fc1_weight']).item(),
            'Hidden Activations': torch.norm(self.gradients['fc1_act']).item(),
            'Layer 2 Weights': torch.norm(self.gradients['fc2_weight']).item(),
            'Output': torch.norm(self.gradients['output']).item()
        }
        
        plt.bar(gradient_norms.keys(), gradient_norms.values())
        plt.title('Gradient Magnitude Through Network')
        plt.ylabel('Gradient Norm')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def demonstrate_backpropagation():
    """Demonstrate backpropagation with visualizations"""
    visualizer = BackpropVisualizer()
    
    # Show the input image
    visualizer.visualize_image_input()
    
    # Visualize gradients
    visualizer.visualize_gradients()
    
    print("The visualization above shows how gradients flow through a neural network during backpropagation.")
    print("Starting from the loss at the output layer, gradients propagate backwards,")
    print("updating weights throughout the network to minimize the error.")

def demonstrate_with_bird_image(image_dir):
    """
    Demonstrate the gradient descent visualizer with bird images
    
    Args:
        image_dir (str): Path to the bird image directory
    """
    try:
        # Get all image files from the directory
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            raise Exception("No image files found in the directory")
            
        # Load and preprocess the first image
        image_path = os.path.join(image_dir, image_files[0])
        img = Image.open(image_path)
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Display original and processed images
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show transformed image (denormalized for visualization)
        img_tensor = transform(img)
        img_display = img_tensor.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        plt.subplot(1, 2, 2)
        plt.imshow(img_display.numpy())
        plt.title('Processed Image (224x224)')
        plt.axis('off')
        
        plt.show()
        
        print("\nImage loaded and preprocessed successfully!")
        print(f"Image shape: {img_tensor.shape}")
        print(f"Image directory: {image_dir}")
        print(f"Found {len(image_files)} images in directory")
        
        # Print some basic image statistics
        print("\nImage Statistics:")
        print(f"Min pixel value: {img_tensor.min():.3f}")
        print(f"Max pixel value: {img_tensor.max():.3f}")
        print(f"Mean pixel value: {img_tensor.mean():.3f}")
        print(f"Std pixel value: {img_tensor.std():.3f}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

def create_interactive_gradient_descent(image_dir):
    """
    Create an interactive, live gradient descent visualization
    """
    # Create interactive widgets
    lr_slider = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0001,
        description='Learning Rate:',
        continuous_update=False
    )
    
    optimizer_dropdown = widgets.Dropdown(
        options=['Adam', 'SGD', 'RMSprop'],
        value='Adam',
        description='Optimizer:'
    )
    
    start_button = widgets.Button(description='Start Training')
    reset_button = widgets.Button(description='Reset')
    
    # Initialize model and data
    def init_model():
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, 200)
        return model
    
    def load_data():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_path = os.path.join(image_dir, image_files[0])
        img = Image.open(image_path)
        return transform(img).unsqueeze(0)
    
    model = init_model()
    img_tensor = load_data()
    losses = []
    gradients = []
    
    def get_optimizer(opt_name, lr):
        if opt_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif opt_name == 'SGD':
            return optim.SGD(model.parameters(), lr=lr)
        else:
            return optim.RMSprop(model.parameters(), lr=lr)
    
    def train_step():
        nonlocal model, losses, gradients
        optimizer = get_optimizer(optimizer_dropdown.value, lr_slider.value)
        
        for step in range(100):  # Run for 100 steps
            optimizer.zero_grad()
            output = model(img_tensor)
            target = torch.tensor([0])
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            grad_norm = sum(p.grad.norm().item() for p in model.parameters())
            losses.append(loss.item())
            gradients.append(grad_norm)
            
            optimizer.step()
            
            # Update plot every 5 steps
            if step % 5 == 0:
                clear_output(wait=True)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                ax1.plot(losses, 'b-')
                ax1.set_title('Loss During Optimization')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(gradients, 'r-')
                ax2.set_title('Gradient Norm')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Gradient Norm')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
                
                # Show current metrics
                with torch.no_grad():
                    prob = torch.nn.functional.softmax(output, dim=1)[0]
                    print(f"Step {step}: Loss = {loss.item():.4f}, "
                          f"Confidence = {prob[0].item()*100:.2f}%")
                
                time.sleep(0.1)  # Small delay to make visualization smoother
    
    def on_start_button_clicked(b):
        losses.clear()
        gradients.clear()
        train_step()
    
    def on_reset_button_clicked(b):
        nonlocal model, losses, gradients
        model = init_model()
        losses.clear()
        gradients.clear()
        clear_output(wait=True)
        plt.close('all')
    
    start_button.on_click(on_start_button_clicked)
    reset_button.on_click(on_reset_button_clicked)
    
    # Display widgets
    controls = widgets.HBox([lr_slider, optimizer_dropdown, start_button, reset_button])
    display(controls)

# Run the visualization
image_dir = "/Users/tanmoy/research/data/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross"
create_interactive_gradient_descent(image_dir)