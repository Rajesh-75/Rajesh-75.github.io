"""
HOG + SVM Image Classifier — built entirely from scratch.

No scikit-learn, no OpenCV, no libsvm/liblinear. Only numpy is used, purely
as an array container / vectorized-arithmetic engine (every algorithmic step
-- gradients, histogram binning, block normalization, subgradient descent --
is implemented explicitly below, not called from a library).

Two training algorithms are implemented, corresponding to the two views of
the SVM objective:
  1. compute_hog(image)   -> Histogram of Oriented Gradients feature vector
  2. LinearSVM (Pegasos)  -> PRIMAL training: stochastic subgradient descent
                             directly on the hinge-loss objective. Short by
                             design -- no dual, no kernel matrix, no KKT
                             bookkeeping. Fast, linear-kernel only.
  3. SMOSVM (Platt's SMO) -> DUAL training: the full Lagrangian-dual
                             algorithm -- box-constrained alpha optimization,
                             KKT-condition checks, kernel support (linear or
                             RBF), bias recovered from the support vectors.
                             This is what makes the kernel trick usable.
  4. OneVsRestSVM         -> multi-class wrapper (works with either trainer)

A synthetic end-to-end demo is included at the bottom so the whole pipeline
can be run and inspected with `python hog_svm_from_scratch.py`.
"""

import numpy as np


# ============================================================
# 1. HOG FEATURE EXTRACTION
# ============================================================

def _compute_gradients(image):
    """
    Central-difference image gradients using the classic Dalal-Triggs
    [-1, 0, 1] kernel (no smoothing -- this simple kernel is what the
    original HOG paper found worked best).

    image: 2D float array (grayscale), shape (H, W)
    returns: (magnitude, orientation_deg) each shape (H, W)
             orientation is UNSIGNED, folded into [0, 180)
    """
    image = image.astype(np.float64)

    gx = np.zeros_like(image)
    gy = np.zeros_like(image)

    # interior pixels: gx = I(x+1) - I(x-1), gy = I(y+1) - I(y-1)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]

    # edge pixels: one-sided difference
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx))   # range (-180, 180]
    orientation = np.mod(orientation, 180.0)       # fold to unsigned [0,180)

    return magnitude, orientation


def _cell_histogram(mag_cell, ori_cell, nbins=9, bin_width=20.0):
    """
    Build one 9-bin (default) unsigned orientation histogram for a single
    cell, using bilinear vote-splitting: each pixel's magnitude is split
    between its two nearest bin centers, weighted by angular distance.
    Bin centers sit at bin_width/2, 3*bin_width/2, ... (10, 30, 50, ...).
    """
    hist = np.zeros(nbins)

    flat_mag = mag_cell.ravel()
    flat_ori = ori_cell.ravel()

    for m, o in zip(flat_mag, flat_ori):
        if m == 0.0:
            continue
        # position in "bin units", center-of-bin-0 at bin_width/2
        pos = o / bin_width - 0.5
        lo = int(np.floor(pos)) % nbins
        hi = (lo + 1) % nbins
        frac = pos - np.floor(pos)          # weight given to the "hi" bin

        hist[lo] += m * (1 - frac)
        hist[hi] += m * frac

    return hist


def compute_hog(image, cell_size=8, block_size=2, nbins=9, eps=1e-6):
    """
    Full HOG descriptor for a single grayscale image.

    image:      2D array, shape (H, W). H and W should be divisible by
                cell_size for a clean grid (crop/pad beforehand otherwise).
    cell_size:  pixels per cell side (e.g. 8 -> 8x8 cells)
    block_size: cells per block side (e.g. 2 -> 2x2 cell blocks)
    nbins:      orientation bins per cell histogram (9 is standard)

    returns: 1D feature vector (concatenated, block-normalized histograms)
    """
    magnitude, orientation = _compute_gradients(image)
    H, W = image.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    # Step 1: per-cell orientation histograms
    cell_hists = np.zeros((n_cells_y, n_cells_x, nbins))
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_hists[cy, cx] = _cell_histogram(
                magnitude[y0:y1, x0:x1], orientation[y0:y1, x0:x1], nbins
            )

    # Step 2: group cells into overlapping blocks, L2-normalize each block
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    if n_blocks_y <= 0 or n_blocks_x <= 0:
        raise ValueError("Image too small for the chosen cell/block size.")

    block_vectors = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hists[by:by + block_size, bx:bx + block_size, :]
            vec = block.ravel()
            norm = np.sqrt(np.sum(vec ** 2) + eps ** 2)
            block_vectors.append(vec / norm)

    return np.concatenate(block_vectors)


# ============================================================
# 2. LINEAR SVM TRAINING -- PEGASOS SUBGRADIENT DESCENT
# ============================================================

class LinearSVM:
    """
    Binary linear SVM trained with Pegasos (Shalev-Shwartz et al., 2007):
    stochastic subgradient descent directly on the primal objective

        f(w) = (lambda/2) * ||w||^2 + (1/n) * sum_i max(0, 1 - y_i <w, x_i>)

    No dual, no kernel matrix, no QP solver -- just a per-example update.
    Labels must be in {-1, +1}. Bias is handled by augmenting every feature
    vector with a constant 1 (a common, simple trick; the bias then lives
    inside w and is only lightly regularized alongside it).
    """

    def __init__(self, lambda_reg=1e-3, n_iters=5000, seed=0):
        self.lambda_reg = lambda_reg
        self.n_iters = n_iters
        self.rng = np.random.RandomState(seed)
        self.w = None  # includes bias as last component

    def fit(self, X, y):
        """
        X: (n_samples, n_features) float array
        y: (n_samples,) array of {-1, +1}
        """
        n_samples, n_features = X.shape
        # augment with bias column
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        w = np.zeros(n_features + 1)

        for t in range(1, self.n_iters + 1):
            i = self.rng.randint(0, n_samples)
            xi, yi = X_aug[i], y[i]
            eta = 1.0 / (self.lambda_reg * t)

            margin = yi * np.dot(w, xi)
            if margin < 1:
                # subgradient of hinge term is -y_i*x_i when margin < 1
                w = (1 - eta * self.lambda_reg) * w + eta * yi * xi
            else:
                # only the regularizer's gradient acts (shrinkage)
                w = (1 - eta * self.lambda_reg) * w

            # optional projection step: keeps ||w|| <= 1/sqrt(lambda),
            # which is where the optimal solution is guaranteed to live.
            # Guard explicitly on norm_w > 0 rather than relying on radius
            # being positive -- this stays safe even if lambda_reg is ever
            # 0, or if the loop/bias-augmentation logic above changes.
            norm_w = np.linalg.norm(w)
            if norm_w > 0:
                radius = 1.0 / np.sqrt(self.lambda_reg)
                if norm_w > radius:
                    w *= radius / norm_w

        self.w = w
        return self

    def decision_function(self, X):
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        return X_aug @ self.w

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class OneVsRestSVM:
    """
    Multi-class wrapper: trains one binary SVM per class (that class vs. all
    others) and predicts the class whose SVM gives the highest decision
    score. Works with either LinearSVM or SMOSVM as the `trainer_cls`.
    """

    def __init__(self, classes, trainer_cls=None, **trainer_kwargs):
        self.classes = list(classes)
        self.trainer_cls = trainer_cls or LinearSVM
        self.trainer_kwargs = trainer_kwargs
        self.models = {}

    def fit(self, X, y):
        for c in self.classes:
            y_bin = np.where(y == c, 1, -1).astype(float)
            model = self.trainer_cls(**self.trainer_kwargs)
            model.fit(X, y_bin)
            self.models[c] = model
        return self

    def predict(self, X):
        scores = np.stack(
            [self.models[c].decision_function(X) for c in self.classes], axis=1
        )
        best = np.argmax(scores, axis=1)
        return np.array([self.classes[i] for i in best])


# ============================================================
# 2b. KERNELIZED SVM TRAINING -- PLATT'S SMO (dual optimization)
# ============================================================
#
# This solves the dual problem directly:
#
#   maximize   sum_i alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)
#   subject to 0 <= alpha_i <= C   for all i
#              sum_i alpha_i y_i = 0
#
# SMO works because that equality constraint means you can never update a
# single alpha_i in isolation -- so instead it repeatedly picks a PAIR
# (alpha_i, alpha_j), solves that 2-variable sub-problem analytically
# (the only step small enough to have a closed form), and repeats until no
# pair violates the KKT conditions by more than `tol`. This is the same
# derivation your dual/KKT tutorial page builds up to; this is the
# algorithm that actually executes it.

def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def rbf_kernel(x1, x2, sigma=1.0):
    diff = x1 - x2
    return np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))


class SMOSVM:
    """
    Kernelized SVM trained with a simplified version of Platt's Sequential
    Minimal Optimization. Solves the dual problem directly via pairwise
    alpha updates -- no external QP solver. Supports 'linear' or 'rbf'.

    This is the "full" training algorithm in the classical sense: it
    produces the Lagrange multipliers (alpha), identifies the support
    vectors (alpha_i > 0), and recovers the bias b from the KKT conditions
    -- everything the dual derivation promises.
    """

    def __init__(self, C=1.0, kernel="linear", sigma=1.0,
                 tol=1e-3, max_passes=10, max_iter=2000):
        self.C = C
        self.kernel_name = kernel
        self.sigma = sigma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter

        if kernel == "linear":
            self._kernel = linear_kernel
        elif kernel == "rbf":
            self._kernel = lambda a, b: rbf_kernel(a, b, sigma)
        else:
            raise ValueError("kernel must be 'linear' or 'rbf'")

    def _compute_kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k_ij = self._kernel(X[i], X[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
        return K

    def _decision_row(self, i, K, alphas, y, b):
        # f(x_i) = sum_k alpha_k y_k K(x_k, x_i) + b
        return np.sum(alphas * y * K[:, i]) + b

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,) in {-1, +1}
        """
        n_samples = X.shape[0]
        alphas = np.zeros(n_samples)
        b = 0.0
        K = self._compute_kernel_matrix(X)

        passes = 0
        it = 0
        while passes < self.max_passes and it < self.max_iter:
            num_changed_alphas = 0

            for i in range(n_samples):
                E_i = self._decision_row(i, K, alphas, y, b) - y[i]

                # KKT violation check for alpha_i
                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or \
                   (y[i] * E_i > self.tol and alphas[i] > 0):

                    # pick j != i at random (textbook heuristic)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    E_j = self._decision_row(j, K, alphas, y, b) - y[j]

                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    # compute box constraints [L, H] for alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue  # not a valid descent direction, skip pair

                    # analytic update for alpha_j, then clip to [L, H]
                    alphas[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # alpha_i moves equal and opposite (keeps sum_i a_i*y_i = 0)
                    alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])

                    # recompute bias from the KKT stationarity condition
                    b1 = (b - E_i
                          - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                          - y[j] * (alphas[j] - alpha_j_old) * K[i, j])
                    b2 = (b - E_j
                          - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                          - y[j] * (alphas[j] - alpha_j_old) * K[j, j])

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            passes = passes + 1 if num_changed_alphas == 0 else 0
            it += 1

        self.alphas = alphas
        self.b = b
        self.X_train = X
        self.y_train = y

        # support vectors are the examples with a non-zero Lagrange multiplier
        sv_mask = alphas > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_alphas_ = alphas[sv_mask]
        self.support_labels_ = y[sv_mask]

        # for a linear kernel, w can be recovered explicitly (w = sum a_i y_i x_i)
        if self.kernel_name == "linear":
            self.w = np.sum(
                (self.support_alphas_ * self.support_labels_)[:, None]
                * self.support_vectors_, axis=0
            )
        else:
            self.w = None  # no explicit w for nonlinear kernels

        return self

    def decision_function(self, X):
        scores = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            s = 0.0
            for alpha_k, y_k, x_k in zip(
                self.support_alphas_, self.support_labels_, self.support_vectors_
            ):
                s += alpha_k * y_k * self._kernel(x_k, x)
            scores[idx] = s + self.b
        return scores

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ============================================================
# 3. END-TO-END DEMO (synthetic images -> HOG -> SVM)
# ============================================================

def _make_synthetic_dataset(n_per_class=40, size=32, seed=0):
    """
    Builds a tiny 3-class toy dataset so the whole pipeline can be run
    without any real image files:
      class 0: a vertical bright bar (strong vertical edges)
      class 1: a horizontal bright bar (strong horizontal edges)
      class 2: a diagonal bright bar (strong diagonal edges)
    Gaussian noise is added so HOG has to do real work, not just detect
    a noise-free template.
    """
    rng = np.random.RandomState(seed)
    images, labels = [], []

    for label in range(3):
        for _ in range(n_per_class):
            img = rng.normal(loc=30, scale=8, size=(size, size))
            thickness = 4
            if label == 0:  # vertical bar
                c = rng.randint(size // 4, 3 * size // 4)
                lo, hi = max(0, c - thickness), min(size, c + thickness)
                img[:, lo:hi] += 180
            elif label == 1:  # horizontal bar
                r = rng.randint(size // 4, 3 * size // 4)
                lo, hi = max(0, r - thickness), min(size, r + thickness)
                img[lo:hi, :] += 180
            else:  # diagonal bar
                for i in range(size):
                    j0 = i - thickness
                    j1 = i + thickness
                    img[i, max(0, j0):min(size, j1)] += 180
            img = np.clip(img, 0, 255)
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    perm = rng.permutation(len(labels))
    return images[perm], labels[perm]


def visualize_predictions(test_imgs, test_labels, preds, class_names,
                           title="SVM Predictions on Test Images",
                           n_show=15, save_path=None):
    """
    Renders a grid of test images with their true vs. predicted class
    printed above each one, green if correct and red if misclassified.
    This is the actual "show me the classified images" output -- printed
    accuracy numbers alone never render anything.

    Requires matplotlib (a plotting library, not an ML library -- it draws
    the pixels the from-scratch code already computed, it does not do any
    of the HOG/SVM work itself).
    """
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(test_imgs))
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.9 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx in range(n_show):
        ax = axes[idx]
        ax.imshow(test_imgs[idx], cmap="gray", vmin=0, vmax=255)
        true_name = class_names[test_labels[idx]]
        pred_name = class_names[preds[idx]]
        correct = test_labels[idx] == preds[idx]
        color = "green" if correct else "red"
        ax.set_title(f"true: {true_name}\npred: {pred_name}",
                     color=color, fontsize=9, pad=6)
        ax.axis("off")

    for idx in range(n_show, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(hspace=0.6, wspace=0.15, top=0.90)

    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"Saved visualization to {save_path}")

    plt.show()


def run_demo():
    print("Building synthetic dataset (3 classes: vertical / horizontal / diagonal bars)...")
    images, labels = _make_synthetic_dataset(n_per_class=40, size=32, seed=1)

    n_train = int(0.75 * len(labels))
    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    print(f"Extracting HOG features for {len(train_imgs)} train / {len(test_imgs)} test images...")
    X_train = np.array([compute_hog(im, cell_size=8, block_size=2) for im in train_imgs])
    X_test = np.array([compute_hog(im, cell_size=8, block_size=2) for im in test_imgs])
    print(f"HOG feature dimensionality: {X_train.shape[1]}")

    print("\n--- Training One-vs-Rest linear SVM via Pegasos (primal) ---")
    clf_pegasos = OneVsRestSVM(
        classes=[0, 1, 2], trainer_cls=LinearSVM,
        lambda_reg=1e-3, n_iters=4000, seed=0,
    )
    clf_pegasos.fit(X_train, train_labels)
    preds = clf_pegasos.predict(X_test)
    accuracy = np.mean(preds == test_labels)
    print(f"Pegasos test accuracy: {accuracy * 100:.1f}%  ({len(test_labels)} test images)")

    print("\n--- Training One-vs-Rest SVM via SMO (dual, RBF kernel) ---")
    clf_smo = OneVsRestSVM(
        classes=[0, 1, 2], trainer_cls=SMOSVM,
        C=1.0, kernel="rbf", sigma=5.0, tol=1e-3, max_passes=5,
    )
    clf_smo.fit(X_train, train_labels)
    preds_smo = clf_smo.predict(X_test)
    accuracy_smo = np.mean(preds_smo == test_labels)
    n_sv = sum(len(m.support_vectors_) for m in clf_smo.models.values())
    print(f"SMO test accuracy: {accuracy_smo * 100:.1f}%  ({len(test_labels)} test images)")
    print(f"Total support vectors across the 3 one-vs-rest models: {n_sv}")

    class_names = {0: "vertical", 1: "horizontal", 2: "diagonal"}
    try:
        visualize_predictions(
            test_imgs, test_labels, preds_smo, class_names,
            title="SMO (RBF kernel) predictions on test images",
            n_show=15, save_path="svm_predictions.png",
        )
    except ImportError:
        print("\n(matplotlib not installed -- run 'pip install matplotlib' "
              "to see the classified images rendered, not just accuracy numbers)")


if __name__ == "__main__":
    run_demo()
