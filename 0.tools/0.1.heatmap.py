# 用热力图的方式可视化一个注意力矩阵
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_matrix(attention_matrix):
    """
    Visualize an n x n attention matrix using a heatmap.
    
    Parameters:
    - attention_matrix: A 2D numpy array or a list of lists of shape (n, n).
    """
    # Ensure attention_matrix is a numpy array for consistency
    attention_matrix = np.array(attention_matrix)
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='Reds', interpolation='nearest')
    
    # Add color bar to the side
    plt.colorbar()
    
    # Add labels and title for clarity
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.title('Attention Matrix Visualization')
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Example usage
    n = 5  # Size of the attention matrix
    attention_matrix = np.random.rand(n, n)  # Generate a random n x n matrix
    print(attention_matrix)
    visualize_attention_matrix(attention_matrix)
