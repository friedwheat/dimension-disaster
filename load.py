"""
load.py — CIFAR-10 data loader for the "Curse of Dimensionality" experiment.

Workflow
--------
1. Read the raw CIFAR-10 pickle batches from *data_dir*.
2. Sub-sample: 200 images / class -> training split (2 000 total);
               50  images / class -> test     split (500 total).
3. Convert each image from the flat 3 072-byte CIFAR layout to an
   OpenCV-compatible (32, 32, 3) BGR array.
4. Build seven flattened feature vectors per image:
       idx  resolution  colour
        0    4 x 4      grayscale
        1    8 x 8      grayscale
        2   12 x 12     grayscale
        3   16 x 16     grayscale
        4   24 x 24     grayscale
        5   32 x 32     grayscale
        6   32 x 32     BGR (colour)
5. Save a visualisation sample with matplotlib.
6. Persist the processed dataset as a pickle file.

Usage
-----
    python load.py --data_dir /path/to/cifar-10-batches-py \
                   --out      features.pkl \
                   --vis_out  sample.png \
                   --seed     42
"""

import argparse
import os
import pickle
import random

import cv2
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_TRAIN_BATCHES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]
CIFAR10_TEST_BATCH = "test_batch"

NUM_CLASSES       = 10
TRAIN_PER_CLASS   = 200   # images sampled per class for training
TEST_PER_CLASS    = 50    # images sampled per class for testing

# (width_or_height, is_colour)  -- defines the 7 feature representations
REPRESENTATIONS = [
    (4,  False),
    (8,  False),
    (12, False),
    (16, False),
    (24, False),
    (32, False),
    (32, True),
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 pickle batch.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to a CIFAR-10 batch file.

    Returns
    -------
    data : np.ndarray, shape (N, 3072), dtype uint8
        Raw pixel rows in CIFAR order (R-plane, G-plane, B-plane).
    labels : list of int
        Class index for each image (0-9).
    """
    with open(filepath, "rb") as fh:
        batch = pickle.load(fh, encoding="bytes")
    data   = batch[b"data"]          # (N, 3072) uint8
    labels = batch[b"labels"]        # list of int
    return data, labels


def load_cifar10(data_dir):
    """Load all CIFAR-10 training and test batches.

    Parameters
    ----------
    data_dir : str
        Directory that contains ``data_batch_1`` ... ``data_batch_5`` and
        ``test_batch``.

    Returns
    -------
    train_data : np.ndarray, shape (50000, 3072)
    train_labels : list of int
    test_data : np.ndarray, shape (10000, 3072)
    test_labels : list of int
    """
    train_data_list   = []
    train_labels_list = []

    for batch_name in CIFAR10_TRAIN_BATCHES:
        path = os.path.join(data_dir, batch_name)
        d, l = load_cifar10_batch(path)
        train_data_list.append(d)
        train_labels_list.extend(l)

    train_data   = np.concatenate(train_data_list, axis=0)   # (50 000, 3072)
    train_labels = train_labels_list

    test_data, test_labels = load_cifar10_batch(
        os.path.join(data_dir, CIFAR10_TEST_BATCH)
    )

    return train_data, train_labels, test_data, list(test_labels)


# ---------------------------------------------------------------------------
# 2. Sample compression
# ---------------------------------------------------------------------------

def subsample(data, labels, n_per_class, seed=42):
    """Randomly draw *n_per_class* images from each of the 10 CIFAR classes.

    Parameters
    ----------
    data : np.ndarray, shape (N, 3072)
    labels : list of int
    n_per_class : int
        Number of images to keep per class.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sub_data : np.ndarray, shape (NUM_CLASSES * n_per_class, 3072)
    sub_labels : list of int, length NUM_CLASSES * n_per_class
    """
    rng = random.Random(seed)
    selected_indices = []

    labels_arr = np.asarray(labels)
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels_arr == cls)[0].tolist()
        chosen = rng.sample(cls_indices, n_per_class)
        selected_indices.extend(chosen)

    selected_indices.sort()  # preserve a reproducible order
    return data[selected_indices], [labels[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# 3. OpenCV adaptation  (CIFAR flat -> BGR image)
# ---------------------------------------------------------------------------

def cifar_row_to_bgr(row):
    """Convert a single 3 072-element CIFAR row to a (32, 32, 3) BGR image.

    CIFAR stores pixels as [R-plane | G-plane | B-plane], each of size
    32 x 32 = 1 024 bytes.  OpenCV expects channels in BGR order.

    Parameters
    ----------
    row : np.ndarray, shape (3072,), dtype uint8

    Returns
    -------
    np.ndarray, shape (32, 32, 3), dtype uint8  -- BGR image
    """
    # Reshape to channel-first (3, 32, 32), then move channels last -> (32, 32, 3)
    img_rgb = row.reshape(3, 32, 32).transpose(1, 2, 0)   # RGB, HWC
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def convert_all_to_bgr(data):
    """Convert every row in *data* to a BGR image.

    Parameters
    ----------
    data : np.ndarray, shape (N, 3072)

    Returns
    -------
    list of N arrays, each shape (32, 32, 3), dtype uint8
    """
    return [cifar_row_to_bgr(row) for row in data]


# ---------------------------------------------------------------------------
# 4. Multi-dimensional feature construction
# ---------------------------------------------------------------------------

def build_features(images_bgr):
    """Build the seven flattened feature matrices for a set of BGR images.

    For each resolution / colour setting in REPRESENTATIONS the function
    resizes every image (using ``cv2.INTER_AREA``), optionally converts to
    grayscale, then flattens and stacks all images into one 2-D array.

    Parameters
    ----------
    images_bgr : list of (32, 32, 3) uint8 BGR arrays

    Returns
    -------
    dict mapping a string key (e.g. ``"4x4_gray"``) to a float32 array of
    shape (N, dim) where dim = size*size  (grayscale) or  size*size*3
    (colour).
    """
    features = {}

    for (size, is_colour) in REPRESENTATIONS:
        rows = []
        for img in images_bgr:
            if size != 32:
                resized = cv2.resize(img, (size, size),
                                     interpolation=cv2.INTER_AREA)
            else:
                resized = img.copy()

            if not is_colour:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            rows.append(resized.flatten().astype(np.float32))

        key = "{}x{}_{}".format(size, size, "bgr" if is_colour else "gray")
        features[key] = np.vstack(rows)   # (N, dim)

    return features


# ---------------------------------------------------------------------------
# 5. Visualisation check
# ---------------------------------------------------------------------------

def visualise_samples(images_bgr, labels, n_show=10,
                      out_path="sample.png", seed=42):
    """Save a grid of random sample images (converted back to RGB for display).

    Parameters
    ----------
    images_bgr : list of (32, 32, 3) BGR arrays
    labels : list of int
        Corresponding class labels.
    n_show : int
        Number of images in the montage.
    out_path : str
        Path for the saved PNG.
    seed : int
        Random seed.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(images_bgr)), min(n_show, len(images_bgr)))

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    cols = min(n_show, 5)
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2, rows * 2),
                             squeeze=False)

    for ax_idx, img_idx in enumerate(indices):
        row_i = ax_idx // cols
        col_i = ax_idx  % cols
        ax = axes[row_i][col_i]

        img_rgb = cv2.cvtColor(images_bgr[img_idx], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(class_names[labels[img_idx]], fontsize=8)
        ax.axis("off")

    # Hide any unused subplots
    for ax_idx in range(len(indices), rows * cols):
        axes[ax_idx // cols][ax_idx % cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print("[vis] Saved sample grid -> {}".format(out_path))


# ---------------------------------------------------------------------------
# 6. Save processed data
# ---------------------------------------------------------------------------

def save_dataset(out_path, train_features, train_labels,
                 test_features, test_labels):
    """Persist the processed dataset to a pickle file.

    The saved dictionary has the structure::

        {
            "train": {
                "4x4_gray":   np.ndarray (2000, 16),
                "8x8_gray":   np.ndarray (2000, 64),
                ...
                "32x32_bgr":  np.ndarray (2000, 3072),
                "labels":     list of int,
            },
            "test": {
                "4x4_gray":   np.ndarray (500, 16),
                ...
                "labels":     list of int,
            },
        }

    Parameters
    ----------
    out_path : str
        Destination pickle file path.
    train_features : dict
        Feature dict for training images.
    train_labels : list of int
        Labels for training images.
    test_features : dict
        Feature dict for test images.
    test_labels : list of int
        Labels for test images.
    """
    dataset = {
        "train": dict(train_features, labels=train_labels),
        "test":  dict(test_features,  labels=test_labels),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("[save] Dataset saved -> {}".format(out_path))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(data_dir, out_path="features.pkl", vis_out="sample.png", seed=42):
    """End-to-end pipeline: load -> subsample -> convert -> features -> save.

    Parameters
    ----------
    data_dir : str
        Path to the CIFAR-10 Python batch directory.
    out_path : str
        Output pickle file for the processed features.
    vis_out : str
        Output PNG for the visualisation check.
    seed : int
        Master random seed.
    """
    # 1. Load raw data
    print("[load] Reading CIFAR-10 batches ...")
    train_raw, train_lbl, test_raw, test_lbl = load_cifar10(data_dir)
    print("       train shape: {}, test shape: {}".format(
        train_raw.shape, test_raw.shape))

    # 2. Sub-sample
    print("[sample] {}/class -> train, {}/class -> test".format(
        TRAIN_PER_CLASS, TEST_PER_CLASS))
    train_sub, train_sub_lbl = subsample(train_raw, train_lbl,
                                         TRAIN_PER_CLASS, seed=seed)
    test_sub,  test_sub_lbl  = subsample(test_raw,  test_lbl,
                                         TEST_PER_CLASS,  seed=seed)
    print("         train: {} images, test: {} images".format(
        len(train_sub_lbl), len(test_sub_lbl)))

    # 3. Convert to BGR
    print("[convert] Reshaping and converting RGB -> BGR ...")
    train_bgr = convert_all_to_bgr(train_sub)
    test_bgr  = convert_all_to_bgr(test_sub)

    # 4. Visualisation (before feature extraction to keep images intact)
    visualise_samples(train_bgr, train_sub_lbl,
                      n_show=10, out_path=vis_out, seed=seed)

    # 5. Build feature representations
    print("[features] Building multi-resolution feature matrices ...")
    train_feat = build_features(train_bgr)
    test_feat  = build_features(test_bgr)

    for key, arr in train_feat.items():
        print("           {}: train {}, test {}".format(
            key, arr.shape, test_feat[key].shape))

    # 6. Save
    save_dataset(out_path, train_feat, train_sub_lbl, test_feat, test_sub_lbl)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Load and preprocess CIFAR-10 for the dimension-disaster experiment."
        )
    )
    parser.add_argument(
        "--data_dir",
        default="cifar-10-batches-py",
        help=(
            "Directory containing the CIFAR-10 Python batch files "
            "(default: cifar-10-batches-py)"
        ),
    )
    parser.add_argument(
        "--out",
        default="features.pkl",
        help="Output pickle file for processed features (default: features.pkl)",
    )
    parser.add_argument(
        "--vis_out",
        default="sample.png",
        help="Output PNG for the sample visualisation (default: sample.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_path=args.out,
         vis_out=args.vis_out, seed=args.seed)
