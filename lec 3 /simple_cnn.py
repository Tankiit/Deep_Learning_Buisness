"""
CNN Layer-by-Layer Visualization

This script:
1. Trains a simple CNN on MNIST (or loads pre-trained model)
2. Visualizes each layer's activations/outputs for a given input
3. Creates detailed visualizations of filters and feature maps
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
import argparse
from collections import OrderedDict


class SimpleCNN(nn.Module):
    """
    A simple CNN for demonstration purposes
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 2: Second Convolutional Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 3: Third Convolutional Layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Final classification layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 3 * 3, num_classes)
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Classification
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


class LayerVisualizer:
    """
    Visualizes the outputs of each layer in a CNN
    """
    def __init__(self, model, output_dir='visualizations'):
        """
        Initialize the visualizer with a model
        
        Args:
            model: PyTorch model to visualize
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Dictionary to store activations
        self.activations = OrderedDict()
        
        # Register hooks to capture activations
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward hooks on all layers to capture activations
        """
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on all modules
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear)):
                module.register_forward_hook(hook_fn(name))
    
    def _process_image(self, image_path=None, tensor=None):
        """
        Process an image for the model
        
        Args:
            image_path: Path to the image file
            tensor: PyTorch tensor (already processed)
            
        Returns:
            PyTorch tensor ready for model input
        """
        if tensor is not None:
            return tensor
        
        if image_path:
            # Load and convert to grayscale
            image = Image.open(image_path).convert('L')
            
            # Transform for MNIST model
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            return transform(image).unsqueeze(0)  # Add batch dimension
        
        # If no image provided, use a sample from MNIST
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=1, shuffle=True
        )
        
        return next(iter(test_loader))[0]  # Get one sample
    
    def _visualize_filters(self, layer_name, layer):
        """
        Visualize filters of a convolutional layer
        
        Args:
            layer_name: Name of the layer
            layer: Layer module
            
        Returns:
            matplotlib figure
        """
        # Get weights
        weights = layer.weight.detach().cpu()
        
        # Number of filters to show
        n_filters = min(16, weights.size(0))
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f'Filters: {layer_name}', fontsize=16)
        
        for i in range(n_filters):
            ax = fig.add_subplot(4, 4, i+1)
            
            # For the first layer, we can visualize directly
            if weights.size(1) == 1:
                filter_img = weights[i, 0]
            else:
                # For deeper layers, take the first channel
                filter_img = weights[i, 0]
            
            # Normalize for better visualization
            filter_min = filter_img.min()
            filter_max = filter_img.max()
            filter_img = (filter_img - filter_min) / (filter_max - filter_min + 1e-8)
            
            ax.imshow(filter_img, cmap='viridis')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def _visualize_activations(self, layer_name, activations):
        """
        Visualize activations of a layer
        
        Args:
            layer_name: Name of the layer
            activations: Activation tensor
            
        Returns:
            matplotlib figure
        """
        # For convolutional layers, show feature maps
        if len(activations.shape) == 4:  # [batch, channels, height, width]
            # Get the first sample in the batch
            act = activations[0]
            
            # Number of channels to show
            n_channels = min(16, act.size(0))
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f'Activations: {layer_name}', fontsize=16)
            
            for i in range(n_channels):
                ax = fig.add_subplot(4, 4, i+1)
                
                feature_map = act[i].cpu().numpy()
                
                # Normalize for better visualization
                if feature_map.max() > feature_map.min():
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                
                ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Channel {i+1}')
                ax.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            return fig
        
        # For fully connected layers, show as a bar chart
        elif len(activations.shape) == 2:  # [batch, features]
            act = activations[0].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(act)), act)
            ax.set_title(f'Activations: {layer_name}')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Activation Value')
            
            return fig
        
        return None
    
    def visualize_all_layers(self, image_path=None, save=True):
        """
        Visualize all layers for a given input
        
        Args:
            image_path: Path to input image (optional)
            save: Whether to save the visualizations
            
        Returns:
            Dictionary of figures
        """
        # Process the image
        input_tensor = self._process_image(image_path)
        
        # Reset activations
        self.activations = OrderedDict()
        
        # Forward pass to get activations
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Store all figures
        figures = OrderedDict()
        
        # Add the input image
        fig, ax = plt.subplots(figsize=(6, 6))
        img = input_tensor[0, 0].cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title('Input Image')
        ax.axis('off')
        figures['input'] = fig
        
        if save:
            fig.savefig(self.output_dir / 'input_image.png')
        
        # Visualize filters and activations for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Visualize filters
                filter_fig = self._visualize_filters(name, module)
                figures[f'{name}_filters'] = filter_fig
                
                if save:
                    filter_fig.savefig(self.output_dir / f'{name}_filters.png')
            
            if name in self.activations:
                # Visualize activations
                act_fig = self._visualize_activations(name, self.activations[name])
                if act_fig:
                    figures[f'{name}_activations'] = act_fig
                    
                    if save:
                        act_fig.savefig(self.output_dir / f'{name}_activations.png')
        
        # Create a summary figure showing progression through the network
        summary_fig = self._create_summary_visualization(input_tensor, predicted_class)
        figures['summary'] = summary_fig
        
        if save:
            summary_fig.savefig(self.output_dir / 'network_summary.png')
        
        print(f"Predicted class: {predicted_class}")
        if save:
            print(f"Visualizations saved to: {self.output_dir}")
        
        return figures
    
    def _create_summary_visualization(self, input_tensor, predicted_class):
        """
        Create a summary visualization showing how the image is transformed
        through the network
        
        Args:
            input_tensor: Input tensor
            predicted_class: Predicted class
            
        Returns:
            matplotlib figure
        """
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('CNN Layer-by-Layer Visualization', fontsize=20)
        
        # Define grid layout
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Input image
        ax_input = fig.add_subplot(gs[0, 0])
        img = input_tensor[0, 0].cpu().numpy()
        ax_input.imshow(img, cmap='gray')
        ax_input.set_title('Input Image')
        ax_input.axis('off')
        
        # After first convolution
        if 'conv1' in self.activations:
            ax_conv1 = fig.add_subplot(gs[0, 1])
            # Show a few channels from the first convolution
            act = self.activations['conv1'][0].cpu().numpy()
            # Combine first 3 channels for RGB-like visualization
            n_channels = min(3, act.shape[0])
            rgb_img = np.zeros((act.shape[1], act.shape[2], 3))
            for i in range(n_channels):
                channel = act[i]
                if channel.max() > channel.min():
                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                rgb_img[:, :, i] = channel
            ax_conv1.imshow(rgb_img)
            ax_conv1.set_title('After Conv1')
            ax_conv1.axis('off')
        
        # After first pooling
        if 'pool1' in self.activations:
            ax_pool1 = fig.add_subplot(gs[0, 2])
            # Show a few channels from the first pooling
            act = self.activations['pool1'][0].cpu().numpy()
            # Combine first 3 channels for RGB-like visualization
            n_channels = min(3, act.shape[0])
            rgb_img = np.zeros((act.shape[1], act.shape[2], 3))
            for i in range(n_channels):
                channel = act[i]
                if channel.max() > channel.min():
                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                rgb_img[:, :, i] = channel
            ax_pool1.imshow(rgb_img)
            ax_pool1.set_title('After Pool1')
            ax_pool1.axis('off')
        
        # After second convolution
        if 'conv2' in self.activations:
            ax_conv2 = fig.add_subplot(gs[1, 0])
            # Show a representative feature map
            act = self.activations['conv2'][0].cpu().numpy()
            representative_idx = act.mean(axis=(1, 2)).argmax()  # Pick the most activated channel
            feature_map = act[representative_idx]
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            ax_conv2.imshow(feature_map, cmap='viridis')
            ax_conv2.set_title('After Conv2\n(Most Active Channel)')
            ax_conv2.axis('off')
        
        # After second pooling
        if 'pool2' in self.activations:
            ax_pool2 = fig.add_subplot(gs[1, 1])
            # Show a representative feature map
            act = self.activations['pool2'][0].cpu().numpy()
            representative_idx = act.mean(axis=(1, 2)).argmax()  # Pick the most activated channel
            feature_map = act[representative_idx]
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            ax_pool2.imshow(feature_map, cmap='viridis')
            ax_pool2.set_title('After Pool2\n(Most Active Channel)')
            ax_pool2.axis('off')
        
        # After third convolution
        if 'conv3' in self.activations:
            ax_conv3 = fig.add_subplot(gs[1, 2])
            # Show a representative feature map
            act = self.activations['conv3'][0].cpu().numpy()
            representative_idx = act.mean(axis=(1, 2)).argmax()  # Pick the most activated channel
            feature_map = act[representative_idx]
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            ax_conv3.imshow(feature_map, cmap='viridis')
            ax_conv3.set_title('After Conv3\n(Most Active Channel)')
            ax_conv3.axis('off')
        
        # After third pooling
        if 'pool3' in self.activations:
            ax_pool3 = fig.add_subplot(gs[2, 0])
            # Show a representative feature map
            act = self.activations['pool3'][0].cpu().numpy()
            representative_idx = act.mean(axis=(1, 2)).argmax()  # Pick the most activated channel
            feature_map = act[representative_idx]
            if feature_map.max() > feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            ax_pool3.imshow(feature_map, cmap='viridis')
            ax_pool3.set_title('After Pool3\n(Most Active Channel)')
            ax_pool3.axis('off')
        
        # Final outputs
        if 'fc' in self.activations:
            ax_output = fig.add_subplot(gs[2, 1:])
            # Create a bar chart of the output probabilities
            output = F.softmax(self.activations['fc'], dim=1)[0].cpu().numpy()
            ax_output.bar(range(len(output)), output)
            ax_output.set_title(f'Final Output (Predicted: {predicted_class})')
            ax_output.set_xlabel('Class')
            ax_output.set_ylabel('Probability')
            ax_output.set_xticks(range(len(output)))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig


def train_model(epochs=5, batch_size=64, learning_rate=0.001, save_path='model.pth'):
    """
    Train a CNN model on MNIST
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        
    Returns:
        Trained model
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='/Users/tanmoy/research/data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SimpleCNN()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss: {running_loss/100:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    return model


def load_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='CNN Layer Visualization')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to load/save model')
    parser.add_argument('--image_path', type=str, default=None, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Train or load model
    if args.train or not os.path.exists(args.model_path):
        print("Training a new model...")
        model = train_model(save_path=args.model_path)
    else:
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)
    
    # Create visualizer
    visualizer = LayerVisualizer(model, output_dir=args.output_dir)
    
    # Visualize all layers
    visualizer.visualize_all_layers(image_path=args.image_path)
    
    print("Done!")


if __name__ == '__main__':
    main()