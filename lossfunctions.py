import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.utils import resample


def load_history_from_json(filename):
    """Load training history data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


# Load the history data for each loss function
history_bce = load_history_from_json('./data/history_bce.json')
history_dice = load_history_from_json('./data/history_dice.json')
history_combined = load_history_from_json('./data/history_combined.json')


def plot_comparison(history_bce, history_dice, history_combined, metric='loss'):
    """Plot a comparison of the given metric for three different training histories."""
    plt.figure(figsize=(12, 6))

    # Plot Loss or another specified metric
    plt.subplot(1, 2, 1)
    plt.plot(history_bce[metric], label='BCE Training')
    plt.plot(history_bce[f'val_{metric}'], label='BCE Validation')

    plt.plot(history_dice[metric], label='Dice Training')
    plt.plot(history_dice[f'val_{metric}'], label='Dice Validation')

    plt.plot(history_combined[metric], label='Combined Training')
    plt.plot(history_combined[f'val_{metric}'], label='Combined Validation')

    plt.title(f'{metric.capitalize()} görbék')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend(loc='best')

    # Plot Accuracy or other metrics if available
    if 'accuracy' in history_bce:
        plt.subplot(1, 2, 2)
        plt.plot(history_bce['accuracy'], label='BCE Training')
        plt.plot(history_bce['val_accuracy'], label='BCE Validation')

        plt.plot(history_dice['accuracy'], label='Dice Training')
        plt.plot(history_dice['val_accuracy'], label='Dice Validation')

        plt.plot(history_combined['accuracy'], label='Combined Training')
        plt.plot(history_combined['val_accuracy'], label='Combined Validation')

        plt.title('Accuracy görbék')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


plot_comparison(history_bce, history_dice, history_combined, metric='loss')


