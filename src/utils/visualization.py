import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_learning_curve(epochs, losses, accuracies, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax2.plot(epochs, accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Level Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_attention_maps(attention_weights, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights.detach().cpu().numpy(), cmap='viridis')
    ax.set_xlabel('Features')
    ax.set_ylabel('Slots')
    ax.set_title('Attention Weights')
    plt.colorbar(im, ax=ax)
    plt.savefig(save_path)
    plt.close()
