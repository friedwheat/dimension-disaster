"""exp2.py – Curse-of-dimensionality experiments on CIFAR-10.

Experiments
-----------
1. Distance analysis
   For each of the 7 feature dimensions compute, over all test samples,
   the ratio  (nearest-neighbour distance) / (mean distance to training set).
   Plot "Distance Ratio vs. Dimension".

2. k-NN classification  (k = 1, 3, 5)
   Compute classification error rate for each dimension and each k.
   Plot one figure with three curves (one per k).

3. Error-case visualisation
   At the highest dimension (3072), find one test sample that is
   misclassified by 1-NN and display it side-by-side with its
   (wrong) nearest-neighbour training sample.
"""

import matplotlib
matplotlib.use('Agg')          # non-interactive backend – safe for all envs

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from exp1 import DIMENSIONS, LABEL_NAMES, load_data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def knn_predict(X_train, y_train, X_test, k):
    """Return predicted labels for X_test using k-NN with Euclidean distance."""
    # dists[i, j] = distance from test sample i to training sample j
    dists = cdist(X_test, X_train, metric='euclidean')  # (n_test, n_train)
    # argpartition with kth=k-1 guarantees the k smallest elements in [:k]
    nn_indices = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    y_pred = np.array([
        np.bincount(y_train[nn_indices[i]], minlength=10).argmax()
        for i in range(len(X_test))
    ])
    return y_pred, dists


def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)


# ---------------------------------------------------------------------------
# Experiment 1 – Distance ratio
# ---------------------------------------------------------------------------

def exp_distance_ratio(data):
    """Compute mean (d_nearest / d_average) for each dimension."""
    ratios = []
    for dim in DIMENSIONS:
        X_train = data[dim]['X_train']
        X_test  = data[dim]['X_test']

        dists = cdist(X_test, X_train, metric='euclidean')  # (n_test, n_train)
        d_nearest = dists.min(axis=1)                        # (n_test,)
        d_average = dists.mean(axis=1)                       # (n_test,)
        ratio = d_nearest / (d_average + 1e-12)              # avoid /0
        ratios.append(ratio.mean())
        print(f'  dim={dim:4d}  mean ratio={ratio.mean():.4f}')

    return ratios


def plot_distance_ratio(ratios):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(DIMENSIONS, ratios, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('Feature Dimension', fontsize=13)
    ax.set_ylabel('Distance Ratio (Nearest / Average)', fontsize=13)
    ax.set_title('Distance Ratio vs. Dimension', fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels(DIMENSIONS, rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig('distance_ratio.png', dpi=150)
    plt.close(fig)
    print('Saved distance_ratio.png')


# ---------------------------------------------------------------------------
# Experiment 2 – k-NN error rates
# ---------------------------------------------------------------------------

def exp_knn(data, k_values=(1, 3, 5)):
    """Return dict  k -> list-of-error-rates (one per dimension)."""
    results = {k: [] for k in k_values}

    for dim in DIMENSIONS:
        X_train = data[dim]['X_train']
        y_train = data[dim]['y_train']
        X_test  = data[dim]['X_test']
        y_test  = data[dim]['y_test']

        # Compute distance matrix once, reuse for all k values
        dists = cdist(X_test, X_train, metric='euclidean')
        max_k = max(k_values)
        # argpartition with kth=max_k-1 guarantees the max_k smallest
        # elements appear in the first max_k columns (unordered)
        nn_indices = np.argpartition(dists, kth=max_k - 1, axis=1)[:, :max_k]
        # Sort each row so that indices are in ascending distance order
        sorted_order = np.argsort(
            dists[np.arange(len(X_test))[:, None], nn_indices], axis=1
        )
        nn_indices_sorted = nn_indices[
            np.arange(len(X_test))[:, None], sorted_order
        ]

        for k in k_values:
            top_k = nn_indices_sorted[:, :k]
            y_pred = np.array([
                np.bincount(y_train[top_k[i]], minlength=10).argmax()
                for i in range(len(X_test))
            ])
            err = error_rate(y_test, y_pred)
            results[k].append(err)
            print(f'  dim={dim:4d}  k={k}  error={err:.4f}')

    return results


def plot_knn(results, k_values=(1, 3, 5)):
    colors  = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, color, marker in zip(k_values, colors, markers):
        ax.plot(DIMENSIONS, results[k],
                marker=marker, linewidth=2, color=color,
                label=f'k = {k}')
    ax.set_xlabel('Feature Dimension', fontsize=13)
    ax.set_ylabel('Classification Error Rate', fontsize=13)
    ax.set_title('k-NN Classification Error vs. Dimension', fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels(DIMENSIONS, rotation=45)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig('knn_error.png', dpi=150)
    plt.close(fig)
    print('Saved knn_error.png')


# ---------------------------------------------------------------------------
# Experiment 3 – Error-case visualisation
# ---------------------------------------------------------------------------

def _to_img(flat):
    """Convert a (3072,) float array back to a (32, 32, 3) uint8 image."""
    img = np.clip(flat, 0.0, 1.0).reshape(3, 32, 32)
    img = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)   # HWC


def exp_error_case(data):
    """Find and display a 1-NN misclassified test sample at 3072 dimensions."""
    dim = 3072
    X_train     = data[dim]['X_train']
    y_train     = data[dim]['y_train']
    X_test      = data[dim]['X_test']
    y_test      = data[dim]['y_test']
    X_train_raw = data['X_train_raw']
    X_test_raw  = data['X_test_raw']

    print('Computing 1-NN distances at dim=3072 ...')
    dists = cdist(X_test, X_train, metric='euclidean')  # (n_test, n_train)
    nn_idx = dists.argmin(axis=1)                        # (n_test,)
    y_pred = y_train[nn_idx]

    # Find misclassified samples
    wrong = np.where(y_test != y_pred)[0]
    if len(wrong) == 0:
        print('No misclassified samples found (unlikely but noted).')
        return

    # Pick the first misclassified sample
    i = wrong[0]
    j = nn_idx[i]

    true_label  = LABEL_NAMES[y_test[i]]
    wrong_label = LABEL_NAMES[y_train[j]]

    print(f'Error case: test index {i}, true={true_label}, '
          f'predicted={wrong_label} (nn train index {j})')

    test_img  = _to_img(X_test_raw[i])
    train_img = _to_img(X_train_raw[j])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(test_img)
    axes[0].set_title(f'Test image\nTrue: {true_label}', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(train_img)
    axes[1].set_title(f'1-NN match (wrong)\nLabel: {wrong_label}', fontsize=11)
    axes[1].axis('off')

    fig.suptitle('1-NN Misclassification Example (dim=3072)', fontsize=12)
    fig.tight_layout()
    fig.savefig('error_case.png', dpi=150)
    plt.close(fig)
    print('Saved error_case.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Loading data ===')
    data = load_data()

    print('\n=== Experiment 1: Distance Ratio ===')
    ratios = exp_distance_ratio(data)
    plot_distance_ratio(ratios)

    print('\n=== Experiment 2: k-NN Classification ===')
    knn_results = exp_knn(data)
    plot_knn(knn_results)

    print('\n=== Experiment 3: Error-case Visualisation ===')
    exp_error_case(data)

    print('\nAll experiments complete.')
