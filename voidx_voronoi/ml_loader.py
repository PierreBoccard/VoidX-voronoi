import numpy as np
import h5py
from typing import Dict, Tuple, Optional

# --------------- I/O ----------------

def load_features_npz(path: str) -> Dict:
    """
    Load features saved via save_features_npz.
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def load_features_hdf5(path: str) -> Dict:
    """
    Load features saved via save_features_hdf5.
    Note: variable-length "cell_vertex_indices" (if present) will be returned as a list.
    """
    out = {}
    with h5py.File(path, "r") as f:
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                out[k] = v[()]
            else:
                # Group (e.g., cell_vertex_indices)
                if k == "cell_vertex_indices":
                    lst = []
                    for i in range(len(v.keys())):
                        ds = v[str(i)][()]
                        lst.append(ds.tolist() if ds.size > 0 else None)
                    out[k] = lst
                else:
                    # Generic group -> dict of datasets
                    out[k] = {name: ds[()] for name, ds in v.items()}
    return out

# --------------- Feature assembly ----------------

def assemble_node_features(features: Dict, log_volume: bool = True, extra_keys: Optional[list] = None) -> np.ndarray:
    """
    Build a per-node numeric feature matrix.
    Defaults: [log(volume), density, neighbor_counts, mean_neighbor_volume]
    """
    vol = features["volumes"].astype(float).copy()
    if log_volume:
        with np.errstate(invalid="ignore"):
            vol = np.log(vol)
    X = [
        vol,
        features["densities"].astype(float),
        features["neighbor_counts"].astype(float),
        features["mean_neighbor_volume"].astype(float),
    ]
    if extra_keys:
        for k in extra_keys:
            X.append(np.asarray(features[k], dtype=float))
    X = np.vstack(X).T  # shape [N, F]
    return X

def build_edge_index_from_csr(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Convert CSR adjacency to edge_index (2, E) with directed edges.
    """
    N = indptr.size - 1
    counts = np.diff(indptr)
    sources = np.repeat(np.arange(N, dtype=np.int64), counts)
    edge_index = np.vstack([sources, indices.astype(np.int64)])
    return edge_index

def build_edge_attr_from_features(features: Dict) -> Optional[np.ndarray]:
    """
    Return per-edge features aligned with edge_index if available (face_areas).
    """
    fa = features.get("face_areas", None)
    if fa is None:
        return None
    return np.asarray(fa, dtype=float)

# --------------- Masks, cleaning, subgraph ----------------

def finite_cell_mask(features: Dict) -> np.ndarray:
    """
    Boolean mask of nodes with finite Voronoi cells (bounded, accepted).
    """
    return np.asarray(features["finite_mask"], dtype=bool)

def nan_safe_impute_and_standardize(X: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Simple imputation: replace NaNs per-column with the mean computed on valid rows.
    Then z-score standardize using mean/std from valid rows.
    Returns transformed X and a dict with fitted stats (mean, std, used_mask).
    """
    if mask is None:
        mask = np.isfinite(X).all(axis=1)
    valid = mask & np.isfinite(X).all(axis=1)
    col_mean = np.nanmean(X[valid], axis=0)
    X_imp = np.where(np.isnan(X), col_mean, X)
    col_std = np.nanstd(X_imp[valid], axis=0)
    col_std[col_std == 0] = 1.0
    X_std = (X_imp - col_mean) / col_std
    stats = {"mean": col_mean, "std": col_std, "valid_mask": valid}
    return X_std, stats

def apply_standardize_with_stats(X: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Apply previously fitted mean/std to new X. NaNs are imputed with fitted mean.
    """
    col_mean = stats["mean"]
    col_std = stats["std"]
    X_imp = np.where(np.isnan(X), col_mean, X)
    return (X_imp - col_mean) / col_std

def subgraph_by_node_mask(edge_index: np.ndarray, node_mask: np.ndarray, edge_attr: Optional[np.ndarray] = None):
    """
    Keep edges where both endpoints are kept. Remap node indices to [0..n_keep-1].
    Returns (edge_index_sub, edge_attr_sub, old_to_new_index)
    """
    N = node_mask.size
    old_to_new = -np.ones(N, dtype=np.int64)
    kept_idx = np.where(node_mask)[0]
    old_to_new[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)

    src = edge_index[0]
    dst = edge_index[1]
    keep_e = node_mask[src] & node_mask[dst]
    src_new = old_to_new[src[keep_e]]
    dst_new = old_to_new[dst[keep_e]]
    edge_index_sub = np.vstack([src_new, dst_new])

    if edge_attr is not None:
        edge_attr_sub = edge_attr[keep_e]
    else:
        edge_attr_sub = None

    return edge_index_sub, edge_attr_sub, old_to_new

# --------------- Quick recipes ----------------

def load_for_sklearn(npz_or_h5_path: str, use_h5: bool = False, extra_keys: Optional[list] = None):
    """
    Returns:
      X: node feature matrix (imputed + standardized)
      mask: boolean mask of finite cells (applied to X)
      meta: dict with raw features and transform stats
    """
    features = load_features_hdf5(npz_or_h5_path) if use_h5 else load_features_npz(npz_or_h5_path)
    mask = finite_cell_mask(features)
    X_raw = assemble_node_features(features, extra_keys=extra_keys)
    # Restrict to finite cells
    X_raw = X_raw[mask]
    X, stats = nan_safe_impute_and_standardize(X_raw)
    meta = {"features": features, "stats": stats}
    return X, mask, meta

def load_for_pyg(npz_or_h5_path: str, use_h5: bool = False, extra_keys: Optional[list] = None, filter_to_finite: bool = True):
    """
    Prepare tensors for PyTorch Geometric.
    Returns:
      x: np.ndarray [N or N_finite, F] (imputed + standardized)
      edge_index: np.ndarray [2, E or E_sub]
      edge_attr: optional np.ndarray [E or E_sub, D] (face area as 1D if present)
      pos: np.ndarray [N or N_finite, 3] original point coordinates
      mapping: dict with 'old_to_new' if subgraphing was applied
      stats: fitted normalization stats
    """
    features = load_features_hdf5(npz_or_h5_path) if use_h5 else load_features_npz(npz_or_h5_path)
    X_raw = assemble_node_features(features, extra_keys=extra_keys)
    edge_index = build_edge_index_from_csr(features["indptr"], features["indices"])
    edge_attr = build_edge_attr_from_features(features)
    pos = np.asarray(features["points"], dtype=float)

    # Normalize features (fit on finite cells)
    finite = finite_cell_mask(features)
    X_norm, stats = nan_safe_impute_and_standardize(X_raw, mask=finite)

    mapping = {}
    if filter_to_finite:
        edge_index, edge_attr, old_to_new = subgraph_by_node_mask(edge_index, finite, edge_attr=edge_attr)
        X_norm = X_norm[finite]
        pos = pos[finite]
        mapping["old_to_new"] = old_to_new

    return X_norm, edge_index, edge_attr, pos, mapping, stats

# --------------- Minimal examples ----------------

if __name__ == "__main__":
    # Example 1: Classical ML (scikit-learn style)
    # Suppose you previously saved: save_features_npz(features, "galaxy_subset_voronoi.npz")
    X, finite_mask, meta = load_for_sklearn("galaxy_subset_voronoi.npz")
    print("Sklearn features:", X.shape)

    # Example unsupervised clustering
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, n_init="auto", random_state=0).fit(X)
        print("KMeans labels shape:", kmeans.labels_.shape)
    except Exception as e:
        print("Install scikit-learn for the clustering example:", e)

    # Example 2: Graph ML (PyTorch Geometric)
    try:
        import torch
        from torch_geometric.data import Data

        Xg, edge_index, edge_attr, pos, mapping, stats = load_for_pyg("galaxy_subset_voronoi.npz", filter_to_finite=True)
        data = Data(
            x=torch.tensor(Xg, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None,
        )
        print(data)
    except Exception as e:
        print("Install torch and torch-geometric for the graph example:", e)