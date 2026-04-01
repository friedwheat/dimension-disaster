import os
import pickle
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def download_cifar10(data_dir="/tmp/cifar10"):
    """Download and extract CIFAR-10 dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        if not os.path.exists(filepath):
            print("Downloading CIFAR-10...")
            urllib.request.urlretrieve(url, filepath)
        print("Extracting CIFAR-10...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(data_dir)

    return extracted_dir


def make_synthetic_cifar10(seed=42):
    """
    Generate synthetic data that mimics two CIFAR-10 classes (airplane vs frog).
    Creates a binary classification problem in 3072-dimensional space (matching
    the 32x32x3 = 3072 pixel features of CIFAR-10).

    The signal lives in a small subspace (first ~8 dimensions) while the
    remaining dimensions are isotropic noise.  This produces a meaningful
    bias-variance tradeoff: at low PCA dimensions the classifier misses some
    signal (bias-dominated), while at very high dimensions the 1-NN is
    increasingly sensitive to the extra noise dimensions (variance-dominated).

    Used as a fallback when the network is unavailable.
    """
    rng = np.random.RandomState(seed)
    n_train_per_class, n_test_per_class = 5000, 1000
    n_features = 3072
    n_signal = 8          # number of truly discriminative dimensions
    signal_sep = 1.0      # mean separation along each signal dimension
    noise_std = 2.5       # std of noise dimensions (large to dominate PCA variance)

    def _make_class(n, label_offset, rng):
        X = rng.randn(n, n_features).astype(np.float32) * noise_std
        # shift the first n_signal dimensions to create signal
        X[:, :n_signal] += label_offset * signal_sep
        # rescale to [0, 255] range (for normalization compatibility)
        X = (X * 20 + 128).clip(0, 255)
        return X

    X_train_a = _make_class(n_train_per_class, -1, rng)
    X_train_b = _make_class(n_train_per_class, +1, rng)
    X_test_a = _make_class(n_test_per_class, -1, rng)
    X_test_b = _make_class(n_test_per_class, +1, rng)

    X_train = np.vstack([X_train_a, X_train_b])
    y_train = np.array([0] * n_train_per_class + [1] * n_train_per_class, dtype=int)
    X_test = np.vstack([X_test_a, X_test_b])
    y_test = np.array([0] * n_test_per_class + [1] * n_test_per_class, dtype=int)

    return X_train, y_train, X_test, y_test


def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], batch[b"labels"]


def load_cifar10(data_dir):
    """Load all CIFAR-10 training and test data."""
    train_data, train_labels = [], []
    for i in range(1, 6):
        filepath = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(filepath)
        train_data.append(data)
        train_labels.extend(labels)

    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


# ── Experiment parameters ──────────────────────────────────────────────────────
SEED = 42
CLASS_A = 0   # airplane
CLASS_B = 6   # frog
M = 40        # number of repeated sampling rounds
N = 150       # training samples per round
N_TEST = 100  # fixed test-set size
DIMS = [1, 2, 4, 8, 16, 32, 64]   # 7 feature dimensions (PCA components)

np.random.seed(SEED)

# ── Load and filter data ───────────────────────────────────────────────────────
try:
    data_dir = download_cifar10()
    X_train_all, y_train_all, X_test_all, y_test_all = load_cifar10(data_dir)

    # Keep only the two selected classes
    train_mask = (y_train_all == CLASS_A) | (y_train_all == CLASS_B)
    X_train_full = X_train_all[train_mask].astype(np.float32) / 255.0
    y_train_full = (y_train_all[train_mask] == CLASS_B).astype(int)  # 0=airplane, 1=frog

    test_mask = (y_test_all == CLASS_A) | (y_test_all == CLASS_B)
    X_test_full = X_test_all[test_mask].astype(np.float32) / 255.0
    y_test_full = (y_test_all[test_mask] == CLASS_B).astype(int)
    print("Loaded real CIFAR-10 data.")
except Exception as e:
    print(f"Could not download CIFAR-10 ({e}). Using synthetic data.")
    X_train_raw, y_train_full, X_test_raw, y_test_full = make_synthetic_cifar10(SEED)
    X_train_full = X_train_raw / 255.0
    X_test_full = X_test_raw / 255.0

# Build a fixed test set of N_TEST samples
rng = np.random.RandomState(SEED)
test_idx = rng.choice(len(X_test_full), N_TEST, replace=False)
X_test_fixed = X_test_full[test_idx]
y_test_fixed = y_test_full[test_idx]

# ── PCA projection (fit on all training data) ──────────────────────────────────
max_dim = max(DIMS)
pca = PCA(n_components=max_dim, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_full)   # (10000, max_dim)
X_test_pca = pca.transform(X_test_fixed)         # (N_TEST, max_dim)

# ── Repeated-sampling experiment ───────────────────────────────────────────────
# all_preds[m, i, d] = prediction for test point i in round m at dimension d
all_preds = np.zeros((M, N_TEST, len(DIMS)), dtype=np.float64)

print(f"Running {M} rounds of 1-NN over {len(DIMS)} dimensions ...")
for m in range(M):
    sample_idx = rng.choice(len(X_train_full), N, replace=False)
    X_tr_pca = X_train_pca[sample_idx]
    y_tr = y_train_full[sample_idx]

    for d_idx, d in enumerate(DIMS):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_tr_pca[:, :d], y_tr)
        all_preds[m, :, d_idx] = knn.predict(X_test_pca[:, :d])

print("Experiment complete. Computing bias-variance decomposition ...")

# ── Bias-Variance-MSE decomposition ───────────────────────────────────────────
# Formulas (averaged over test points):
#   MSE_p   = mean_i( mean_m( (ŷ_m(x_i) - y_i)^2 ) )
#   Var_p   = mean_i( mean_m( (ŷ_m(x_i) - ȳ(x_i))^2 ) )  where ȳ = mean over m
#   Bias²_p = MSE_p - Var_p
MSE = np.zeros(len(DIMS))
Var = np.zeros(len(DIMS))
Bias2 = np.zeros(len(DIMS))

for d_idx in range(len(DIMS)):
    preds = all_preds[:, :, d_idx]           # (M, N_TEST)
    mean_pred = preds.mean(axis=0)            # (N_TEST,)

    mse_per_test = np.mean((preds - y_test_fixed[np.newaxis, :]) ** 2, axis=0)
    var_per_test = np.mean((preds - mean_pred[np.newaxis, :]) ** 2, axis=0)

    MSE[d_idx] = mse_per_test.mean()
    Var[d_idx] = var_per_test.mean()
    Bias2[d_idx] = MSE[d_idx] - Var[d_idx]

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'Dim':>5}  {'MSE':>8}  {'Variance':>10}  {'Sq.Bias':>10}")
print("-" * 42)
for d_idx, d in enumerate(DIMS):
    print(f"{d:>5}  {MSE[d_idx]:>8.5f}  {Var[d_idx]:>10.5f}  {Bias2[d_idx]:>10.5f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
plt.plot(DIMS, MSE, "b-o", label="MSE")
plt.plot(DIMS, Var, "r-s", label="Variance")
plt.plot(DIMS, Bias2, "g-^", label="Sq. Bias")
plt.xlabel("Feature Dimension (PCA components)")
plt.ylabel("Error")
plt.title(
    "Bias–Variance Decomposition vs. Feature Dimension\n"
    f"(1-NN, CIFAR-10 Airplane vs Frog, M={M} rounds, N={N} training samples)"
)
plt.legend()
plt.grid(True)
plt.xticks(DIMS)
plt.tight_layout()
plt.savefig("exp3_result.png", dpi=150)
print("\nPlot saved to exp3_result.png")
plt.show()
