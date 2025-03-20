import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def plot_attention(attention_weights, tokens, title="Attention Weights"):
    """Plot attention weights as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="YlGnBu", 
                vmin=0, 
                vmax=np.max(attention_weights))
    plt.title(title, fontsize=15)
    plt.xlabel('Keys', fontsize=12)
    plt.ylabel('Queries', fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def simple_self_attention_example():
    """
    Demonstrating self-attention with a simple example:
    "The animal didn't cross the street because it was too tired."
    """
    # Tokenize the sentence
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."]
    n_tokens = len(tokens)
    
    # Simplified embedding size
    d_model = 64
    
    # Normally these would come from trained matrices, but for demonstration
    # we'll define simple embeddings and transformations
    
    # Create random embeddings for our tokens
    np.random.seed(42)  # For reproducible results
    embeddings = np.random.randn(n_tokens, d_model)
    
    # Create random projection matrices for query, key, value
    W_query = np.random.randn(d_model, d_model)
    W_key = np.random.randn(d_model, d_model)
    W_value = np.random.randn(d_model, d_model)
    
    # Step 1: Project embeddings to get query, key, value vectors
    queries = embeddings.dot(W_query)
    keys = embeddings.dot(W_key)
    values = embeddings.dot(W_value)
    
    # Step 2: Calculate attention scores (Q·K^T)
    attention_scores = np.matmul(queries, keys.T)
    
    # Step 3: Scale scores
    d_k = keys.shape[1]
    attention_scores = attention_scores / np.sqrt(d_k)
    
    # Step 4: Apply softmax to get attention weights
    attention_weights = softmax(attention_scores)
    
    # Step 5: Compute weighted sum of values
    attention_output = np.matmul(attention_weights, values)
    
    # Visualize attention weights
    attention_fig = plot_attention(attention_weights, tokens)
    
    # For our example, modify some weights to illustrate the relationship
    # between "it" and "animal" (this is artificial just for visualization)
    manual_weights = attention_weights.copy()
    
    # Set high attention from "it" to "animal"
    it_idx = tokens.index("it")
    animal_idx = tokens.index("animal")
    
    # Reset the row for "it"
    manual_weights[it_idx] = 0.01 * np.ones(n_tokens)
    
    # Set specific attention weights
    manual_weights[it_idx, animal_idx] = 0.85  # "it" attends strongly to "animal"
    manual_weights[it_idx, tokens.index("street")] = 0.05
    manual_weights[it_idx, tokens.index("tired")] = 0.08
    # Normalize
    manual_weights[it_idx] = manual_weights[it_idx] / np.sum(manual_weights[it_idx])
    
    # Visualize the manual attention weights
    manual_fig = plot_attention(manual_weights, tokens, 
                          "Attention Weights (with strong 'it' → 'animal' connection)")
    
    return {
        "tokens": tokens,
        "attention_weights": attention_weights,
        "manual_weights": manual_weights,
        "attention_output": attention_output
    }

def multi_head_attention_example(num_heads=3):
    """
    Demonstrating multi-head attention with our example sentence
    """
    # Tokenize the sentence
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired", "."]
    n_tokens = len(tokens)
    
    # Simplified embedding size
    d_model = 64
    head_dim = d_model // num_heads
    
    np.random.seed(42)  # For reproducible results
    
    # Create random embeddings
    embeddings = np.random.randn(n_tokens, d_model)
    
    # Initialize arrays to store attention weights for each head
    all_attention_weights = []
    all_outputs = []
    
    for h in range(num_heads):
        # Create random projection matrices for this head
        W_query = np.random.randn(d_model, head_dim)
        W_key = np.random.randn(d_model, head_dim)
        W_value = np.random.randn(d_model, head_dim)
        
        # Project embeddings to get query, key, value vectors for this head
        queries = embeddings.dot(W_query)
        keys = embeddings.dot(W_key)
        values = embeddings.dot(W_value)
        
        # Calculate attention scores
        attention_scores = np.matmul(queries, keys.T)
        
        # Scale scores
        attention_scores = attention_scores / np.sqrt(head_dim)
        
        # Apply softmax
        attention_weights = softmax(attention_scores)
        
        # For demonstration, we'll manually modify each head to show different patterns
        if h == 0:
            # Head 1: Focus on adjacent words (local context)
            for i in range(n_tokens):
                attention_weights[i] = 0.01 * np.ones(n_tokens)
                # Attend to self and neighbors
                for j in range(max(0, i-1), min(n_tokens, i+2)):
                    attention_weights[i, j] = 0.3
                # Normalize
                attention_weights[i] = attention_weights[i] / np.sum(attention_weights[i])
                
        elif h == 1:
            # Head 2: Subject-verb relationships
            # For simplicity, just highlight a few key relationships
            subject_idxs = [tokens.index("animal")]
            verb_idxs = [tokens.index("didn't"), tokens.index("cross")]
            
            for subj in subject_idxs:
                for verb in verb_idxs:
                    attention_weights[subj, verb] = 0.5
                    attention_weights[verb, subj] = 0.5
            
            # Normalize
            for i in range(n_tokens):
                attention_weights[i] = attention_weights[i] / np.sum(attention_weights[i])
                
        elif h == 2:
            # Head 3: Coreference (pronouns)
            # Focus on "it" -> "animal" relationship
            it_idx = tokens.index("it")
            animal_idx = tokens.index("animal")
            
            attention_weights[it_idx] = 0.01 * np.ones(n_tokens)
            attention_weights[it_idx, animal_idx] = 0.8
            
            # Normalize
            attention_weights[it_idx] = attention_weights[it_idx] / np.sum(attention_weights[it_idx])
        
        # Compute output for this head
        head_output = np.matmul(attention_weights, values)
        all_outputs.append(head_output)
        all_attention_weights.append(attention_weights)
        
        # Visualize attention for this head
        title = f"Head {h+1} Attention Pattern"
        _ = plot_attention(attention_weights, tokens, title)
    
    # Concatenate outputs from all heads
    multi_head_output = np.concatenate(all_outputs, axis=1)
    
    # Typically there would be a final projection here
    
    return {
        "tokens": tokens,
        "head_attention_weights": all_attention_weights,
        "multi_head_output": multi_head_output
    }

# Run example
single_head_results = simple_self_attention_example()
multi_head_results = multi_head_attention_example(num_heads=3)

print("Single-head attention complete.")
print("Multi-head attention complete.")
print("\nView the visualizations to see the attention patterns.")

# Save the figures to files
plt.figure(1)  # Access the first figure (basic attention)
plt.savefig('single_head_attention.png')

plt.figure(2)  # Access the second figure (manual weights)
plt.savefig('single_head_manual_attention.png')

# Save multi-head attention figures
for i in range(3):  # Assuming 3 heads as in your example
    plt.figure(i+3)  # The multi-head figures start from index 3
    plt.savefig(f'multi_head_attention_{i+1}.png')

print("All visualizations have been saved to the current directory.")

plt.show()  # Show plots after saving