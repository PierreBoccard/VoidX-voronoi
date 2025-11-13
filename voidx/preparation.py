import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Optional, Tuple
from numpy.lib.format import open_memmap
import builtins

class GalaxySyntheticDataPreparer:
    def __init__(
        self,
        box_size_mpc: float,
        use_periodic_boundaries: bool = True,
        N_neighbours: int = 20,
        rng: Optional[np.random.Generator] = None,
        chunk_size_query: int = int(2e5),
        rows_per_chunk_tensor: int = int(1e5),
    ) -> None:
        self.box_size_mpc = float(box_size_mpc)
        self.use_pbc = bool(use_periodic_boundaries)
        self.N = int(N_neighbours)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.chunk_size_query = int(chunk_size_query)
        self.rows_per_chunk_tensor = int(rows_per_chunk_tensor)

    # ---------- Sampling helpers ----------
    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def sample_shell_galaxies(
        self,
        n: int,
        center: np.ndarray,
        radius: float,
        thickness: float,
        jitter_scale: float = 0.0,
    ) -> np.ndarray:
        directions = self._normalize_rows(self.rng.normal(size=(n, 3)))
        radial_offset = self.rng.normal(loc=0.0, scale=thickness / 2.0, size=n) if thickness > 0 else np.zeros(n)
        radii = radius + radial_offset
        if thickness > 0.0:
            lower = float(np.maximum(radius - thickness, 0.0))
            upper = radius + thickness
            radii = np.clip(radii, lower, upper)
        positions = directions * radii[:, None]
        if jitter_scale > 0.0:
            positions += self.rng.normal(scale=jitter_scale * radius, size=(n, 3))
        return positions + center[None, :]

    def sample_void_galaxies(
        self,
        n: int,
        center: np.ndarray,
        core_radius: float,
        beta: float = 0.0,
    ) -> np.ndarray:
        if n == 0:
            return np.empty((0, 3), dtype=float)
        beta_eff = float(np.maximum(beta, -2.99))
        directions = self._normalize_rows(self.rng.normal(size=(n, 3)))
        u = self.rng.uniform(size=n)
        radii = core_radius * np.power(u, 1.0 / (3.0 + beta_eff))
        return directions * radii[:, None] + center[None, :]

    @staticmethod
    def _wrap_to_box(pts: np.ndarray, box: float) -> np.ndarray:
        return ((pts + box / 2.0) % box) - box / 2.0

    def sample_thomas_background(
        self,
        parent_density: float,
        offspring_mean: float,
        cluster_sigma: float,
    ) -> np.ndarray:
        box_size = self.box_size_mpc
        volume = box_size ** 3
        expected_parents = parent_density * volume
        n_parents = self.rng.poisson(expected_parents)
        parents = self.rng.uniform(low=-box_size/2.0, high=box_size/2.0, size=(n_parents, 3)) if n_parents > 0 else np.empty((0,3))
        if n_parents == 0:
            return np.empty((0, 3), dtype=float)
        offspring_counts = self.rng.poisson(lam=offspring_mean, size=n_parents)
        total_offspring = int(offspring_counts.sum())
        if total_offspring == 0:
            return np.empty((0, 3), dtype=float)
        pts = np.empty((total_offspring, 3), dtype=float)
        idx = 0
        for p, k in zip(parents, offspring_counts):
            if k == 0:
                continue
            pts[idx:idx+k, :] = p[None, :] + self.rng.normal(scale=cluster_sigma, size=(k, 3))
            idx += k
        return self._wrap_to_box(pts, box_size) if self.use_pbc else pts[(pts >= -box_size/2.0).all(1) & (pts <= box_size/2.0).all(1)]

    @staticmethod
    def _minimal_image_vec(d: np.ndarray, box: float) -> np.ndarray:
        return d - box * np.round(d / box)

    def sample_nonoverlapping_voids(
        self,
        n: int,
        rmin: float,
        rmax: float,
        max_attempts_per_void: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        centers: list[np.ndarray] = []
        radii: list[float] = []
        attempts = 0
        while len(centers) < n and attempts < max_attempts_per_void * max(1, int(n)):
            R = float(self.rng.uniform(rmin, rmax))
            c = self.rng.uniform(low=-self.box_size_mpc/2.0, high=self.box_size_mpc/2.0, size=3)
            ok = True
            for cj, Rj in zip(centers, radii):
                d = c - cj
                if self.use_pbc:
                    d = self._minimal_image_vec(d, self.box_size_mpc)
                if np.dot(d, d) < (R + Rj) ** 2:
                    ok = False
                    break
            if ok:
                centers.append(c)
                radii.append(R)
            attempts += 1
        return np.array(centers, dtype=float), np.array(radii, dtype=float)

    # ---------- Catalogue builder ----------
    def build_catalogue(
        self,
        num_voids: int,
        void_radius_min: float,
        void_radius_max: float,
        per_void_shell_galaxies: int,
        per_void_void_galaxies: int,
        shell_thickness: float,
        core_radius_fraction: float,
        void_radial_bias_beta: float,
        directional_jitter: float = 0.0,
        background_parent_density: float = 5.0e-3,
        background_offspring_mean: int = 600,
        background_cluster_sigma: float = 1.6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        bg = self.sample_thomas_background(
            parent_density=background_parent_density,
            offspring_mean=background_offspring_mean,
            cluster_sigma=background_cluster_sigma,
        )
        n_bg_raw = bg.shape[0]
        centers, radii = self.sample_nonoverlapping_voids(
            n=int(num_voids), rmin=float(void_radius_min), rmax=float(void_radius_max), max_attempts_per_void=50
        )
        core_radii = core_radius_fraction * radii
        # remove background inside cores
        if bg.size and core_radii.size:
            if self.use_pbc:
                dx = bg[:, None, :] - centers[None, :, :]
                dx -= self.box_size_mpc * np.round(dx / self.box_size_mpc)
                dist2 = np.sum(dx * dx, axis=2)
                inside_any = (dist2 < (core_radii[None, :] ** 2)).any(axis=1)
            else:
                from scipy.spatial.distance import cdist as _cdist
                dmat = _cdist(bg, centers)
                inside_any = (dmat < core_radii[None, :]).any(axis=1)
            bg = bg[~inside_any]
        n_bg_kept = bg.shape[0]
        shell_list = []
        void_list = []
        for c, R, Rc in zip(centers, radii, core_radii):
            shell_list.append(self.sample_shell_galaxies(
                n=per_void_shell_galaxies, center=c, radius=float(R), thickness=float(shell_thickness), jitter_scale=float(directional_jitter)
            ))
            void_list.append(self.sample_void_galaxies(
                n=per_void_void_galaxies, center=c, core_radius=float(Rc), beta=float(void_radial_bias_beta)
            ))
        shell_positions = np.vstack(shell_list) if shell_list else np.empty((0, 3))
        void_positions = np.vstack(void_list) if void_list else np.empty((0, 3))
        positions = np.vstack([bg, shell_positions, void_positions])
        labels = np.array([
            "background"
        ] * bg.shape[0] + [
            "shell"
        ] * shell_positions.shape[0] + [
            "void"
        ] * void_positions.shape[0])
        return positions, labels, centers, radii, n_bg_raw, n_bg_kept

    # ---------- k-NN computations ----------
    def _periodic_query_chunk(
        self,
        tree: cKDTree,
        pts: np.ndarray,
        k: int,
        box: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Periodic-boundary query fallback for older SciPy without boxsize support.

        Performs queries for 27 periodic shifts of the query points and
        deduplicates neighbors per row, keeping the closest k results.

        Returns (dists: float64 [mchunk,k], indices: int64 [mchunk,k]).
        The first column corresponds to the self-distance (0) and a placeholder
        index which callers may overwrite with the true global index if needed.
        """
        # Generate 27 shift vectors
        shifts = (
            np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"))
            .reshape(3, -1)
            .T
            * box
        )
        all_d, all_i = [], []
        for s in shifts:
            # Query with and without workers depending on SciPy support
            try:
                d, ind = tree.query(pts + s, k=k, workers=-1)
            except TypeError:
                d, ind = tree.query(pts + s, k=k)
            all_d.append(d)
            all_i.append(ind)

        d_cat = np.concatenate(all_d, axis=1)
        i_cat = np.concatenate(all_i, axis=1)

        mchunk = pts.shape[0]
        d_out = np.empty((mchunk, k), dtype=np.float64)
        i_out = np.empty((mchunk, k), dtype=np.int64)
        for i in range(mchunk):
            # self first (caller may overwrite indices later)
            d_out[i, 0] = 0.0
            i_out[i, 0] = 0
            order = np.argsort(d_cat[i])
            seen = set()
            w = 1
            for j in order:
                cand = int(i_cat[i, j])
                if cand in seen:
                    continue
                seen.add(cand)
                i_out[i, w] = cand
                d_out[i, w] = d_cat[i, j]
                w += 1
                if w == k:
                    break
            # Pad if needed (should be rare)
            while w < k:
                i_out[i, w] = i_out[i, w - 1]
                d_out[i, w] = d_out[i, w - 1] + 1e-9
                w += 1
        return d_out, i_out

    def compute_neighbors(
        self,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first-N neighbor distances and indices for all galaxies in chunks.
        Returns (idx: int64 [M,N], dists: float32 [M,N]).
        """
        N = self.N
        M = positions.shape[0]
        positions64 = positions.astype(np.float64, copy=False)
        tree = cKDTree(positions64)
        idxs = np.empty((M, N), dtype=np.int64)
        dists = np.empty((M, N), dtype=np.float32)

        k_total = N + 1
        for start in range(0, M, self.chunk_size_query):
            end = min(start + self.chunk_size_query, M)
            pts = positions64[start:end]
            if self.use_pbc:
                try:
                    d, ind = tree.query(pts, k=k_total, workers=-1, boxsize=self.box_size_mpc)
                except TypeError:
                    d, ind = self._periodic_query_chunk(tree, pts, k=k_total, box=self.box_size_mpc)
                    ind[:, 0] = np.arange(start, end, dtype=np.int64)
            else:
                try:
                    d, ind = tree.query(pts, k=k_total, workers=-1)
                except TypeError:
                    d, ind = tree.query(pts, k=k_total)
            dists[start:end] = d[:, 1:].astype(np.float32)
            idxs[start:end] = ind[:, 1:].astype(np.int64)
        return idxs, dists

    def build_neighbor_tensor_memmap(
        self,
        positions: np.ndarray,
        first_neighbor_idx: np.ndarray,
        out_path: Path,
    ) -> Path:
        """Build (M,N,N) neighbor-of-neighbor distance tensor into a memmap file and return its path."""
        N = self.N
        M = positions.shape[0]
        positions64 = positions.astype(np.float64, copy=False)
        tree = cKDTree(positions64)
        X_mm = open_memmap(str(out_path), mode='w+', dtype=np.float32, shape=(M, N, N))
        rows = max(1, int(self.rows_per_chunk_tensor))
        for start in range(0, M, rows):
            end = min(start + rows, M)
            neigh_chunk = first_neighbor_idx[start:end]  # (R, N)
            neigh_unique = np.unique(neigh_chunk.ravel())
            pts_u = positions64[neigh_unique]
            if self.use_pbc:
                try:
                    d_u, _ = tree.query(pts_u, k=N + 1, boxsize=self.box_size_mpc)
                except TypeError:
                    d_u, _ = tree.query(pts_u, k=N + 1)
            else:
                d_u, _ = tree.query(pts_u, k=N + 1)
            idx_in_unique = np.searchsorted(neigh_unique, neigh_chunk.ravel()).reshape(end - start, N)
            d_collect = d_u[idx_in_unique][:, :, 1:].astype(np.float32)
            X_mm[start:end] = d_collect
        del X_mm  # flush
        return out_path

    def build_neighbor_tensor_array(
        self,
        positions: np.ndarray,
        first_neighbor_idx: np.ndarray,
    ) -> np.ndarray:
        """Build (M,N,N) neighbor-of-neighbor distance tensor in memory and return it.

        Warning: This allocates ~ M * N * N * 4 bytes. Use only for small-enough sizes.
        """
        N = self.N
        M = positions.shape[0]
        positions64 = positions.astype(np.float64, copy=False)
        tree = cKDTree(positions64)
        X = np.empty((M, N, N), dtype=np.float32)
        rows = max(1, int(self.rows_per_chunk_tensor))
        for start in range(0, M, rows):
            end = min(start + rows, M)
            neigh_chunk = first_neighbor_idx[start:end]
            neigh_unique = np.unique(neigh_chunk.ravel())
            pts_u = positions64[neigh_unique]
            if self.use_pbc:
                try:
                    d_u, _ = tree.query(pts_u, k=N + 1, boxsize=self.box_size_mpc)
                except TypeError:
                    d_u, _ = tree.query(pts_u, k=N + 1)
            else:
                d_u, _ = tree.query(pts_u, k=N + 1)
            idx_in_unique = np.searchsorted(neigh_unique, neigh_chunk.ravel()).reshape(end - start, N)
            d_collect = d_u[idx_in_unique][:, :, 1:].astype(np.float32)
            X[start:end] = d_collect
        return X

    # ---------- Density profiles ----------
    def compute_density_profiles(
        self,
        positions: np.ndarray,
        void_centers: np.ndarray,
        void_radii: np.ndarray,
        rmax_factor: float = 5.0,
        nbins: int = 40,
    ):
        bin_edges = np.linspace(0.0, rmax_factor, nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        four_thirds_pi = (4.0 / 3.0) * np.pi
        shell_factor = four_thirds_pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        volume_box = self.box_size_mpc ** 3
        nbar = positions.shape[0] / volume_box
        N_voids = void_centers.shape[0]
        density_profiles = np.zeros((N_voids, nbins), dtype=np.float64)
        delta_profiles = np.zeros_like(density_profiles)
        def minimal_image(dx: np.ndarray, box: float) -> np.ndarray:
            return dx - box * np.round(dx / box) if self.use_pbc else dx
        for i in range(N_voids):
            c = void_centers[i]
            Rv = float(void_radii[i])
            dx = positions - c[None, :]
            dx = minimal_image(dx, self.box_size_mpc)
            r = np.linalg.norm(dx, axis=1)
            x = r / Rv
            mask = x <= rmax_factor
            if not np.any(mask):
                continue
            counts, _ = np.histogram(x[mask], bins=bin_edges)
            Vshell = shell_factor * (Rv ** 3)
            rho = counts / Vshell
            delta = rho / nbar - 1.0
            density_profiles[i] = rho
            delta_profiles[i] = delta
        return bin_edges, bin_centers, density_profiles, delta_profiles, nbar

    # ---------- Save helper ----------
    @staticmethod
    def save_knn_outputs(
        knn_file: Path,
        positions: np.ndarray,
        labels: np.ndarray,
        first_neighbor_dist: np.ndarray,
        N: int,
        void_centers: np.ndarray,
        X_knn: Optional[np.ndarray] = None,
        memmap_path: Optional[Path] = None,
    ) -> None:
        payload = {
            'positions': positions.astype(np.float32),
            'membership': (labels == 'void').astype(np.int8),
            'first_neighbor_distances': first_neighbor_dist.astype(np.float32),
            'N': np.int32(N),
            'void_centers': void_centers.astype(np.float32),
        }
        if X_knn is not None:
            payload['tensor_distance'] = X_knn.astype(np.float32)
        if memmap_path is not None:
            payload['tensor_distance_memmap_path'] = np.array(str(memmap_path))
        np.savez_compressed(knn_file, **payload)
        print(f"Saved k-NN outputs to {knn_file}")


# --------- Density profile utilities extracted from notebook ---------
def stack_density_stats(
    density_profiles: np.ndarray,
    delta_profiles: np.ndarray,
):
    """Compute stacked statistics for density and delta profiles across voids.

    Returns tuple of 10 arrays:
      (stack_mean_density, stack_median_density, stack_p16_density, stack_p84_density,
       stack_mean_delta,   stack_median_delta,   stack_p16_delta,   stack_p84_delta)
    """
    stack_mean_density = density_profiles.mean(axis=0)
    stack_median_density = np.median(density_profiles, axis=0)
    stack_p16_density, stack_p84_density = np.percentile(density_profiles, [16, 84], axis=0)

    stack_mean_delta = delta_profiles.mean(axis=0)
    stack_median_delta = np.median(delta_profiles, axis=0)
    stack_p16_delta, stack_p84_delta = np.percentile(delta_profiles, [16, 84], axis=0)

    return (
        stack_mean_density,
        stack_median_density,
        stack_p16_density,
        stack_p84_density,
        stack_mean_delta,
        stack_median_delta,
        stack_p16_delta,
        stack_p84_delta,
    )


def plot_density_profiles(
    bin_centers: np.ndarray,
    density_profiles: np.ndarray,
    delta_profiles: np.ndarray,
    stack_mean_density: np.ndarray,
    stack_p16_density: np.ndarray,
    stack_p84_density: np.ndarray,
    stack_mean_delta: np.ndarray,
    stack_p16_delta: np.ndarray,
    stack_p84_delta: np.ndarray,
    max_voids_to_plot: int = 10,
    save_dir: Optional[Path] = None,
):
    """Plot stacked density/delta and a grid of individual profiles.

    Note: Imports matplotlib lazily to avoid mandatory dependency at import time.
    """
    import matplotlib.pyplot as plt

    # Stacked profiles
    plt.figure(figsize=(8, 4.5))
    plt.plot(bin_centers, stack_mean_density, label="Mean density", color="#1f77b4", lw=2)
    plt.fill_between(bin_centers, stack_p16_density, stack_p84_density, color="#1f77b4", alpha=0.2, label="16–84%")
    plt.xlabel("R / Rv")
    plt.ylabel("Density ρ [gal / (Mpc/h)^3]")
    plt.title("Stacked void density profile ρ(R/Rv)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir : 
        save_path = save_dir / "stacked_density_profile.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved stacked density profile plot to: {save_path}")
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(bin_centers, stack_mean_delta, label="Mean δ", color="#d62728", lw=2)
    plt.fill_between(bin_centers, stack_p16_delta, stack_p84_delta, color="#d62728", alpha=0.2, label="16–84%")
    plt.axhline(0.0, color="k", lw=1, alpha=0.5)
    plt.xlabel("R / Rv")
    plt.ylabel("Density contrast δ = ρ/ρ̄ − 1")
    plt.title("Stacked void density contrast δ(R/Rv)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir :
        save_path = save_dir / "stacked_delta_profile.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved stacked delta profile plot to: {save_path}")
    plt.show()

    # Individual profiles (first few voids)
    profile_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    n_plot = builtins.min(max_voids_to_plot, int(density_profiles.shape[0]))
    if n_plot > 0:
        plt.figure(figsize=(12, 4.5))
        plt.subplot(1, 2, 1)
        for i in range(n_plot):
            plt.plot(
                bin_centers,
                density_profiles[i],
                label=f"Void {i}",
                color=profile_colors[i % len(profile_colors)],
                lw=1.5,
                alpha=0.8,
            )
        plt.xlabel("R / Rv")
        plt.ylabel("Density ρ [gal / (Mpc/h)^3]")
        plt.title("Individual void density profiles (first few voids)")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        for i in range(n_plot):
            plt.plot(
                bin_centers,
                delta_profiles[i],
                label=f"Void {i}",
                color=profile_colors[i % len(profile_colors)],
                lw=1.5,
                alpha=0.8,
            )
        plt.axhline(0.0, color="k", lw=1, alpha=0.5)
        plt.xlabel("R / Rv")
        plt.ylabel("Density contrast δ = ρ/ρ̄ − 1")
        plt.title("Individual void density contrast (first few voids)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_dir :
            save_path = save_dir / "individual_void_profiles.png"
            plt.savefig(save_path, dpi=150)
            print(f"Saved individual void profiles plot to: {save_path}")
        plt.show()


def save_density_profiles_npz(
    out_path: Path,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    density_profiles: np.ndarray,
    delta_profiles: np.ndarray,
    stack_mean_density: np.ndarray,
    stack_median_density: np.ndarray,
    stack_p16_density: np.ndarray,
    stack_p84_density: np.ndarray,
    stack_mean_delta: np.ndarray,
    stack_median_delta: np.ndarray,
    stack_p16_delta: np.ndarray,
    stack_p84_delta: np.ndarray,
    nbar: float,
    rmax_factor: float,
    nbins: int,
):
    payload = dict(
        bin_edges=bin_edges.astype(np.float32),
        bin_centers=bin_centers.astype(np.float32),
        density_profiles=density_profiles.astype(np.float32),
        delta_profiles=delta_profiles.astype(np.float32),
        stack_mean_density=stack_mean_density.astype(np.float32),
        stack_median_density=stack_median_density.astype(np.float32),
        stack_p16_density=stack_p16_density.astype(np.float32),
        stack_p84_density=stack_p84_density.astype(np.float32),
        stack_mean_delta=stack_mean_delta.astype(np.float32),
        stack_median_delta=stack_median_delta.astype(np.float32),
        stack_p16_delta=stack_p16_delta.astype(np.float32),
        stack_p84_delta=stack_p84_delta.astype(np.float32),
        nbar=np.float32(nbar),
        rmax_factor=np.float32(rmax_factor),
        nbins=np.int32(nbins),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved per-void profiles to: {out_path}")


# --------- Background pruning and k-NN recompute utilities ---------
def select_top_background_by_mean_knn(
    first_neighbor_dist: np.ndarray,
    labels: np.ndarray,
    n_remove: int,
    K_MEAN: int = 20,
    background_label: str = "background",
):
    """Select indices of background galaxies with highest mean distance over first K neighbors.

    Returns (rm_indices, mean_over_k, k_cols) where rm_indices are absolute indices in the
    current catalogue.
    """
    bg_mask = (labels == background_label)
    bg_indices = np.flatnonzero(bg_mask)
    if bg_indices.size == 0:
        raise ValueError("No background galaxies found for removal.")
    k_cols = builtins.min(int(K_MEAN), first_neighbor_dist.shape[1])
    mean_over_k = first_neighbor_dist[:, :k_cols].mean(axis=1)
    k_remove = int(builtins.min(int(n_remove), bg_indices.size))
    idx_in_bg = np.argpartition(mean_over_k[bg_indices], -k_remove)[-k_remove:]
    rm_indices = bg_indices[idx_in_bg]
    # Sort descending by mean distance for determinism
    rm_indices = rm_indices[np.argsort(mean_over_k[rm_indices])[::-1]]
    return rm_indices, mean_over_k, k_cols


def remove_points(
    positions: np.ndarray,
    labels: np.ndarray,
    rm_indices: np.ndarray,
):
    """Remove points at rm_indices and return updated arrays and keep mask."""
    keep_mask = np.ones(positions.shape[0], dtype=bool)
    keep_mask[rm_indices] = False
    return positions[keep_mask], labels[keep_mask], keep_mask


def recompute_first_neighbors(
    positions: np.ndarray,
    N: int,
    box_size_mpc: float,
    use_periodic_boundaries: bool = True,
    chunk_size_query: int = int(2e5),
    rng: Optional[np.random.Generator] = None,
):
    """Wrapper over class-based compute to (idx, dists) after any catalogue change."""
    return compute_first_neighbors(
        positions=positions,
        N=N,
        box_size_mpc=box_size_mpc,
        use_periodic_boundaries=use_periodic_boundaries,
        chunk_size_query=chunk_size_query,
        rng=rng,
    )


def build_neighbor_tensor_to_memmap(
    positions: np.ndarray,
    first_neighbor_idx: np.ndarray,
    N: int,
    box_size_mpc: float,
    use_periodic_boundaries: bool,
    out_path: Path,
    rows_per_chunk_tensor: int = int(1e5),
    rng: Optional[np.random.Generator] = None,
):
    """Create (M,N,N) neighbor-of-neighbor distance tensor into out_dir and return its path."""
    # out_dir.mkdir(parents=True, exist_ok=True)
    M = positions.shape[0]
    # out_path = out_dir / f"tensor_distance_M{M}_N{int(N)}.npy"
    prep = GalaxySyntheticDataPreparer(
        box_size_mpc=box_size_mpc,
        use_periodic_boundaries=use_periodic_boundaries,
        N_neighbours=N,
        rng=rng,
        rows_per_chunk_tensor=rows_per_chunk_tensor,
    )
    return prep.build_neighbor_tensor_memmap(positions, first_neighbor_idx, out_path)


def plot_knn_curves_by_class_and_topk(
    first_neighbor_dist: np.ndarray,
    labels: np.ndarray,
    k_cols: int,
    n_remove: int,
    title: str = "k-NN distance curves",
    save_dir: Optional[Path] = None,
):
    """Plot top-k galaxies by mean-over-k and class means of k-NN distances."""
    import matplotlib.pyplot as plt
    import numpy as np
    neighbor_order = np.arange(1, first_neighbor_dist.shape[1] + 1)
    mean_over_k = first_neighbor_dist[:, :k_cols].mean(axis=1)
    k_top = builtins.min(int(n_remove), first_neighbor_dist.shape[0])
    top_idx = np.argpartition(mean_over_k, -k_top)[-k_top:]
    top_idx = top_idx[np.argsort(mean_over_k[top_idx])[::-1]]

    plt.figure(figsize=(8, 4.5))
    for j, gi in enumerate(top_idx):
        if j == 0:
            plt.plot(neighbor_order, first_neighbor_dist[gi], color="tab:red", alpha=0.6, lw=1.5, label=f"Top-{k_top} galaxies")
        else:
            plt.plot(neighbor_order, first_neighbor_dist[gi], color="tab:red", alpha=0.3, lw=1.0)

    # Class means
    classes = ["background", "void", "shell"]
    colors = {"shell": "#1f77b4", "void": "#010000", "background": "#2ca02c"}
    for cls in classes:
        mask = (labels == cls)
        if np.any(mask):
            plt.plot(neighbor_order, first_neighbor_dist[mask].mean(axis=0), label=f"{cls.capitalize()} mean", color=colors.get(cls, None), linewidth=3)

    plt.xlabel("Neighbor index")
    plt.ylabel("Distance [Mpc/h]")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir :
        save_path = save_dir / "knn_distance_curves.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved k-NN distance curves plot to: {save_path}")
    plt.show()


# --------- Convenience top-level functions ---------
def compute_neighbor_tensor_direct(
    positions: np.ndarray,
    first_neighbor_idx: np.ndarray,
    box_size_mpc: float,
    use_periodic_boundaries: bool = True,
    rows_per_chunk_tensor: int = int(1e5),
) -> np.ndarray:
    """Compute (M, N, N) neighbor-of-neighbor distance tensor directly in memory.

    For each galaxy i (0..M-1), for each of its N neighbors j, we compute distances
    from neighbor j to its own N nearest neighbors, using periodic boundaries if requested.

    Returns float32 tensor X with shape (M, N, N).
    """
    positions64 = positions.astype(np.float64, copy=False)
    M, N = first_neighbor_idx.shape
    X = np.empty((M, N, N), dtype=np.float32)
    box = float(box_size_mpc)

    def minimal_image(delta: np.ndarray) -> np.ndarray:
        if not use_periodic_boundaries:
            return delta
        return delta - box * np.round(delta / box)

    rows = max(1, int(rows_per_chunk_tensor))
    for start in range(0, M, rows):
        end = min(M, start + rows)
        idx_block = first_neighbor_idx[start:end]  # (B, N)
        B = end - start
        for b in range(B):
            nbrs = idx_block[b]              # (N,)
            nbr_pos = positions64[nbrs]      # (N,3)
            for j in range(N):
                nb_j = nbrs[j]
                nn = first_neighbor_idx[nb_j]        # (N,)
                nn_pos = positions64[nn]             # (N,3)
                dvec = nn_pos - nbr_pos[j]           # (N,3)
                dvec = minimal_image(dvec)
                X[start + b, j, :] = np.linalg.norm(dvec, axis=1)
    return X

def compute_first_neighbors(
    positions: np.ndarray,
    N: int,
    box_size_mpc: float,
    use_periodic_boundaries: bool = True,
    chunk_size_query: int = int(2e5),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute indices and distances to the first N neighbors for all points.

    Returns (idx: int64 [M,N], dists: float32 [M,N]).
    """
    prep = GalaxySyntheticDataPreparer(
        box_size_mpc=box_size_mpc,
        use_periodic_boundaries=use_periodic_boundaries,
        N_neighbours=N,
        rng=rng,
        chunk_size_query=chunk_size_query,
    )
    return prep.compute_neighbors(positions)


def compute_neighbor_distances_and_tensor(
    positions: np.ndarray,
    N: int,
    box_size_mpc: float,
    out_path: Path,
    use_periodic_boundaries: bool = True,
    chunk_size_query: int = int(2e5),
    rows_per_chunk_tensor: int = int(1e5),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Compute first-N neighbor distances/indices and build (M,N,N) neighbor tensor to disk.

    Returns (idx: int64 [M,N], dists: float32 [M,N], tensor_path: Path).
    """
    prep = GalaxySyntheticDataPreparer(
        box_size_mpc=box_size_mpc,
        use_periodic_boundaries=use_periodic_boundaries,
        N_neighbours=N,
        rng=rng,
        chunk_size_query=chunk_size_query,
        rows_per_chunk_tensor=rows_per_chunk_tensor,
    )
    idx, dists = prep.compute_neighbors(positions)
    tensor_path = prep.build_neighbor_tensor_memmap(positions, idx, out_path)
    return idx, dists, tensor_path


def compute_knn_features(
    positions: np.ndarray,
    N: int,
    box_size_mpc: float,
    use_periodic_boundaries: bool = True,
    chunk_size_query: int = int(2e5),
    rows_per_chunk_tensor: int = int(1e5),
    compute_neighbor_cube: bool = False,
    cube_ram_threshold_gib: float = 1.25,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute first-N neighbor features and optionally an in-memory (M,N,N) neighbor-of-neighbor cube.

    Returns (first_neighbor_idx: int64 [M,N], first_neighbor_dist: float32 [M,N], X_knn or None).
    If compute_neighbor_cube is True but estimated size exceeds cube_ram_threshold_gib, X_knn=None and a
    message is printed recommending a memmap workflow.
    """
    prep = GalaxySyntheticDataPreparer(
        box_size_mpc=box_size_mpc,
        use_periodic_boundaries=use_periodic_boundaries,
        N_neighbours=N,
        rng=rng,
        chunk_size_query=chunk_size_query,
        rows_per_chunk_tensor=rows_per_chunk_tensor,
    )
    idx, dists = prep.compute_neighbors(positions)

    X_knn: Optional[np.ndarray] = None
    if compute_neighbor_cube:
        M = positions.shape[0]
        approx_bytes = float(M) * float(N) * float(N) * 4.0
        approx_gib = approx_bytes / (1024 ** 3)
        if approx_gib > float(cube_ram_threshold_gib):
            print(
                f"Skipping neighbor-of-neighbor cube: ~{approx_gib:.2f} GiB required (>{cube_ram_threshold_gib:.2f} GiB threshold). "
                "Set compute_neighbor_cube=False, reduce N, or use memmap via build_neighbor_tensor_to_memmap()."
            )
        else:
            X_knn = prep.build_neighbor_tensor_array(positions, idx)
    return idx, dists, X_knn