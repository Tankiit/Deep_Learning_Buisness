import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Create a function to visualize the convolution process
def visualize_convolution(input_image, filter_kernel, title, annotation_pos=None):
    # Pad the input for convolution
    padded_input = np.pad(input_image, ((1, 1), (1, 1)), mode='constant')
    
    # Calculate output dimensions
    output_height = input_image.shape[0]
    output_width = input_image.shape[1]
    
    # Initialize output
    output = np.zeros((output_height, output_width))
    
    # Apply convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the region of interest
            roi = padded_input[i:i+3, j:j+3]
            # Apply the filter (element-wise multiplication and sum)
            output[i, j] = np.sum(roi * filter_kernel)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Create a custom colormap that goes from white to black
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "black"])
    
    # Define the grid layout
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # 1. Show the input image
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(input_image, cmap=cmap)
    ax1.set_title("Input Image")
    # Add grid
    for i in range(input_image.shape[0] + 1):
        ax1.axhline(i - 0.5, color='red', linewidth=0.5)
    for j in range(input_image.shape[1] + 1):
        ax1.axvline(j - 0.5, color='red', linewidth=0.5)
    # Add pixel values
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            ax1.text(j, i, f"{input_image[i, j]:.0f}", 
                     ha="center", va="center", color="blue")
    
    # 2. Show the filter
    ax2 = plt.subplot(gs[0, 1])
    im2 = ax2.imshow(filter_kernel, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title("Filter Kernel (3×3)")
    # Add grid
    for i in range(filter_kernel.shape[0] + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(filter_kernel.shape[1] + 1):
        ax2.axvline(j - 0.5, color='black', linewidth=0.5)
    # Add filter values
    for i in range(filter_kernel.shape[0]):
        for j in range(filter_kernel.shape[1]):
            ax2.text(j, i, f"{filter_kernel[i, j]:.1f}", 
                     ha="center", va="center", color="black")
    
    # 3. Show the convolution calculation example
    if annotation_pos is not None:
        i, j = annotation_pos
        ax3 = plt.subplot(gs[0, 2])
        ax3.axis('off')
        ax3.set_title("Convolution Calculation Example")
        
        # Extract the region for the example
        roi = padded_input[i:i+3, j:j+3]
        
        # Create the calculation text
        calc_text = f"Output[{i},{j}] = Sum(Input × Filter)\n\n"
        calc_terms = []
        
        for ii in range(3):
            for jj in range(3):
                calc_terms.append(f"({roi[ii,jj]:.0f} × {filter_kernel[ii,jj]:.1f})")
        
        calc_text += " + ".join(calc_terms) + "\n\n"
        calc_text += f"= {np.sum(roi * filter_kernel):.1f}"
        
        ax3.text(0.1, 0.5, calc_text, fontsize=10, va='center')
    
    # 4. Show the output image
    ax4 = plt.subplot(gs[1, 0:2])
    # Normalize output for better visualization
    normalized_output = output.copy()
    if np.max(np.abs(output)) > 0:
        normalized_output = output / np.max(np.abs(output))
    
    im4 = ax4.imshow(normalized_output, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title("Output Feature Map")
    # Add grid
    for i in range(output.shape[0] + 1):
        ax4.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(output.shape[1] + 1):
        ax4.axvline(j - 0.5, color='black', linewidth=0.5)
    # Add output values
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            ax4.text(j, i, f"{output[i, j]:.1f}", 
                     ha="center", va="center", color="black")
    
    # 5. Feature identification explanation
    ax5 = plt.subplot(gs[1, 2])
    ax5.axis('off')
    ax5.set_title("Feature Identification")
    
    # Add explanation text based on filter type
    if "Horizontal Edge" in title:
        explanation = (
            "• Strong positive (red) values indicate transitions\n"
            "  from dark to light moving down\n\n"
            "• Strong negative (blue) values indicate transitions\n"
            "  from light to dark moving down\n\n"
            "• The filter responds most strongly to horizontal edges\n"
            "  and ignores vertical edges"
        )
    elif "Diagonal Edge" in title:
        explanation = (
            "• Strong positive (red) values indicate transitions\n"
            "  from dark to light along the main diagonal\n\n"
            "• Strong negative (blue) values indicate transitions\n"
            "  from light to dark along the main diagonal\n\n"
            "• The filter responds to edges running from\n"
            "  top-left to bottom-right"
        )
    else:  # Texture pattern
        explanation = (
            "• Strong positive (red) values indicate areas\n"
            "  matching the target texture pattern\n\n"
            "• Strong negative (blue) values indicate areas\n"
            "  with opposite pattern\n\n"
            "• This filter detects the specific texture pattern\n"
            "  defined in the filter kernel"
        )
    
    ax5.text(0.1, 0.5, explanation, fontsize=10, va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, output

# 1. Horizontal Edge Filter
horizontal_edge_filter = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

# 2. Diagonal Edge Filter (top-left to bottom-right)
diagonal_edge_filter = np.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
])

# 3. Texture Pattern Filter (checkerboard pattern)
texture_filter = np.array([
    [1, -1, 1],
    [-1, 1, -1],
    [1, -1, 1]
])

# Create a sample input image (6×6 grid with a horizontal edge)
horizontal_edge_image = np.array([
    [50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50],
    [150, 150, 150, 150, 150, 150],
    [150, 150, 150, 150, 150, 150],
    [150, 150, 150, 150, 150, 150]
])

# Create a sample input image with diagonal edge
diagonal_edge_image = np.array([
    [50, 50, 50, 150, 150, 150],
    [50, 50, 50, 150, 150, 150],
    [50, 50, 100, 150, 150, 150],
    [50, 50, 150, 150, 150, 150],
    [50, 150, 150, 150, 150, 150],
    [150, 150, 150, 150, 150, 150]
])

# Create a sample input image with checkerboard pattern
texture_image = np.array([
    [50, 150, 50, 150, 50, 150],
    [150, 50, 150, 50, 150, 50],
    [50, 150, 50, 150, 50, 150],
    [150, 50, 150, 50, 150, 50],
    [50, 150, 50, 150, 50, 150],
    [150, 50, 150, 50, 150, 50]
])

# Visualize horizontal edge filter
fig1, _ = visualize_convolution(
    horizontal_edge_image, 
    horizontal_edge_filter,
    "Horizontal Edge Detection Filter",
    annotation_pos=(2, 2)  # Position for sample calculation
)

# Visualize diagonal edge filter
fig2, _ = visualize_convolution(
    diagonal_edge_image, 
    diagonal_edge_filter,
    "Diagonal Edge Detection Filter (Top-left to Bottom-right)",
    annotation_pos=(2, 2)
)

# Visualize texture filter
fig3, _ = visualize_convolution(
    texture_image, 
    texture_filter,
    "Checkerboard Texture Pattern Filter",
    annotation_pos=(2, 2)
)

plt.show()