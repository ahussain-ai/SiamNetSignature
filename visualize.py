import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import tensorflow as tf


# Plot the images with labels
def plot_batch(images1, images2, labels):
    batch_size = images1.shape[0]
    fig, axes = plt.subplots(10, 2, figsize=(10, 10 * 2.5))

    for i in range(10):
        # Plot the first image in the pair
        ax = axes[i, 0]
        ax.imshow(tf.squeeze(images1[i]))
        ax.set_title(f"Pair {i+1} - Image 1\nLabel: {labels[i].numpy()}")
        ax.axis('off')

        # Plot the second image in the pair
        ax = axes[i, 1]
        ax.imshow(tf.squeeze(images2[i]))
        ax.set_title(f"Pair {i+1} - Image 2\nLabel: {labels[i].numpy()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()