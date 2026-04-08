import os
import pickle
import tarfile
import urllib.request

import numpy as np
from sklearn.decomposition import PCA

# The 7 feature dimensions used across experiments
DIMENSIONS = [16, 32, 64, 128, 256, 512, 3072]

# CIFAR-10 label names
LABEL_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

_CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CIFAR10_DIR = 'cifar-10-batches-py'


def _download_cifar10():
    """Download and extract CIFAR-10 if not already present."""
    if os.path.isdir(_CIFAR10_DIR):
        return
    fname = 'cifar-10-python.tar.gz'
    if not os.path.isfile(fname):
        print(f'Downloading CIFAR-10 from {_CIFAR10_URL} ...')
        try:
            urllib.request.urlretrieve(_CIFAR10_URL, fname)
        except Exception as exc:
            raise RuntimeError(
                f'Could not download CIFAR-10: {exc}\n'
                'Please download cifar-10-python.tar.gz manually from '
                f'{_CIFAR10_URL} and place it in the current directory.'
            ) from exc
        print('Download complete.')
    print('Extracting CIFAR-10 ...')
    with tarfile.open(fname, 'r:gz') as tar:
        tar.extractall()
    print('Extraction complete.')


def _generate_synthetic(n_train=5000, n_test=1000, random_state=42):
    """Generate synthetic CIFAR-10-shaped data for offline testing.

    Returns arrays with the same shape and dtype as the real CIFAR-10 subset
    used by ``load_data``, but with random pixel values and labels.
    This is **not** intended for real experiments – it exists only to allow
    the code to be exercised when the CIFAR-10 dataset is unavailable.
    """
    rng = np.random.default_rng(random_state)
    X_train = rng.random((n_train, 3072), dtype=np.float32)
    y_train = rng.integers(0, 10, size=n_train, dtype=np.int32)
    X_test  = rng.random((n_test,  3072), dtype=np.float32)
    y_test  = rng.integers(0, 10, size=n_test,  dtype=np.int32)
    return X_train, y_train, X_test, y_test


def _load_batch(path):
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='bytes')
    images = entry[b'data'].astype(np.float32) / 255.0  # shape (N, 3072)
    labels = np.array(entry[b'labels'], dtype=np.int32)
    return images, labels


def load_cifar10_raw():
    """Return raw (flattened, normalised) CIFAR-10 arrays.

    Falls back to synthetic data when the dataset cannot be downloaded
    (e.g. in an offline CI environment).

    Returns
    -------
    X_train : ndarray, shape (50000, 3072)  [or synthetic size]
    y_train : ndarray, shape (50000,)
    X_test  : ndarray, shape (10000, 3072)  [or synthetic size]
    y_test  : ndarray, shape (10000,)
    """
    try:
        _download_cifar10()
    except RuntimeError as exc:
        print(f'WARNING: {exc}')
        print('Falling back to synthetic random data (for testing only).')
        return _generate_synthetic()

    train_images, train_labels = [], []
    for i in range(1, 6):
        path = os.path.join(_CIFAR10_DIR, f'data_batch_{i}')
        imgs, lbls = _load_batch(path)
        train_images.append(imgs)
        train_labels.append(lbls)
    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    X_test, y_test = _load_batch(os.path.join(_CIFAR10_DIR, 'test_batch'))
    return X_train, y_train, X_test, y_test


def load_data(n_train=5000, n_test=1000, random_state=42):
    """Load CIFAR-10 and extract PCA features at each dimension in DIMENSIONS.

    To keep computation manageable a random subset of the dataset is used.

    Parameters
    ----------
    n_train : int
        Number of training samples to use.
    n_test : int
        Number of test samples to use.
    random_state : int
        Seed for reproducible subsampling.

    Returns
    -------
    data : dict
        Keys are the dimension values from ``DIMENSIONS``.
        Each value is a dict with keys:
            'X_train'  - ndarray (n_train, dim)
            'X_test'   - ndarray (n_test,  dim)
            'y_train'  - ndarray (n_train,)
            'y_test'   - ndarray (n_test,)
        Additionally, the top-level dict contains:
            'y_train'      - training labels aligned with the feature arrays
            'y_test'       - test labels aligned with the feature arrays
            'X_train_raw'  - raw 3072-d training images (for visualisation)
            'X_test_raw'   - raw 3072-d test images (for visualisation)
    """
    X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10_raw()

    rng = np.random.default_rng(random_state)
    n_train = min(n_train, len(X_train_full))
    n_test  = min(n_test,  len(X_test_full))
    train_idx = rng.choice(len(X_train_full), size=n_train, replace=False)
    test_idx  = rng.choice(len(X_test_full),  size=n_test,  replace=False)

    X_train_raw = X_train_full[train_idx]
    y_train     = y_train_full[train_idx]
    X_test_raw  = X_test_full[test_idx]
    y_test      = y_test_full[test_idx]

    # Centre the training data; apply the same shift to test data
    mean = X_train_raw.mean(axis=0)
    X_train_c = X_train_raw - mean
    X_test_c  = X_test_raw  - mean

    # Fit PCA on training data with enough components for all dimensions
    max_pca_dim = max(d for d in DIMENSIONS if d < 3072)
    print(f'Fitting PCA with {max_pca_dim} components on {n_train} samples ...')
    pca = PCA(n_components=max_pca_dim, random_state=random_state)
    pca.fit(X_train_c)
    print('PCA fitting done.')

    # Project to all PCA dimensions at once and slice
    Z_train_full = pca.transform(X_train_c)  # (n_train, max_pca_dim)
    Z_test_full  = pca.transform(X_test_c)   # (n_test,  max_pca_dim)

    data = {
        'y_train':      y_train,
        'y_test':       y_test,
        'X_train_raw':  X_train_raw,
        'X_test_raw':   X_test_raw,
    }

    for dim in DIMENSIONS:
        if dim == 3072:
            data[dim] = {
                'X_train': X_train_raw,
                'X_test':  X_test_raw,
                'y_train': y_train,
                'y_test':  y_test,
            }
        else:
            data[dim] = {
                'X_train': Z_train_full[:, :dim],
                'X_test':  Z_test_full[:,  :dim],
                'y_train': y_train,
                'y_test':  y_test,
            }

    return data


if __name__ == '__main__':
    data = load_data()
    print('Dimensions available:', [k for k in data if isinstance(k, int)])
    for dim in DIMENSIONS:
        print(f'  dim={dim}: X_train={data[dim]["X_train"].shape}, '
              f'X_test={data[dim]["X_test"].shape}')
