import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Optional HDBSCAN support
try:
    import hdbscan  # type: ignore
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False


class GalaxyClusteringAnalyzer:
    """
    A comprehensive 3D galaxy clustering analyzer with automatic optimization.

    Features:
    - Multiple clustering algorithms (K-Means, DBSCAN, optional HDBSCAN)
    - Automatic parameter optimization (find_optimal_k, estimate_natural_clusters)
    - Quality metrics calculation
    - Optional visualization (controlled by verbose)
    - Clean, reusable interface
    """

    def __init__(self, data: Union[str, Path, np.ndarray], verbose: bool = True):
        """
        Initialize the analyzer.

        Args:
            data: Either a path to a whitespace-delimited file with at least 3 columns,
                  or a NumPy array of shape (n_points, 3).
            verbose: If True, prints progress and shows plots.
        """
        self.verbose = verbose
        if isinstance(data, (str, Path)):
            self.data_path = Path(data)
            self.X = self._load_data(self.data_path)
        else:
            self.data_path = None
            self.X = self._validate_array(data)

        self.n_points = len(self.X)
        if self.verbose:
            src = self.data_path.name if self.data_path else "ndarray"
            print(f"Loaded {self.n_points} galaxy coordinates from {src}")

    @staticmethod
    def _validate_array(arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError("data must be a NumPy array when not providing a file path.")
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("data array must be of shape (n_points, >=3).")
        return np.asarray(arr[:, :3], dtype=float)

    @staticmethod
    def _load_data(path: Path) -> np.ndarray:
        """Load 3D coordinates from whitespace-delimited file."""
        coordinates: List[Tuple[float, float, float]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        x, y, z = map(float, parts[:3])
                        coordinates.append((x, y, z))
                    except ValueError:
                        continue
        except Exception as e:
            raise ValueError(f"Error reading data file: {e}")

        if not coordinates:
            raise ValueError("No valid 3D coordinates found in the file")
        return np.array(coordinates, dtype=float)

    def _calculate_metrics(self, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics."""
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = int(np.sum(labels == -1))

        metrics: Dict[str, object] = {
            "n_points": int(len(labels)),
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "noise_ratio": float(n_noise) / float(len(labels)),
            "cluster_sizes": {},
        }

        # Count cluster sizes
        for label in unique_labels:
            metrics["cluster_sizes"][int(label)] = int(np.sum(labels == label))

        # Quality metrics (for at least 2 clusters; exclude noise)
        if n_clusters >= 2:
            try:
                mask = labels != -1
                if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
                    metrics["silhouette_score"] = float(
                        silhouette_score(self.X[mask], labels[mask])
                    )
                    metrics["calinski_harabasz_score"] = float(
                        calinski_harabasz_score(self.X[mask], labels[mask])
                    )
                    metrics["davies_bouldin_score"] = float(
                        davies_bouldin_score(self.X[mask], labels[mask])
                    )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not calculate quality metrics: {e}")

        return metrics

    def _plot_3d_clusters(
        self,
        labels: np.ndarray,
        centers: Optional[np.ndarray] = None,
        title: str = "3D Galaxy Clustering",
        save_path: Optional[Union[str, Path]] = None,
        show: Optional[bool] = None,
    ):
        """Plot 3D clustering results."""
        if show is None:
            show = self.verbose
        if not show and save_path is None:
            return

        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_labels))))

        fig = go.Figure(data=[])

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                fig.add_trace(go.Scatter3d(
                    x=self.X[mask, 0],
                    y=self.X[mask, 1],
                    z=self.X[mask, 2],
                    mode='markers',
                    marker=dict(size=3, color='lightgray', opacity=0.6),
                    name='Noise'
                ))

            else:
                fig.add_trace(go.Scatter3d(
                    x=self.X[mask, 0],
                    y=self.X[mask, 1],
                    z=self.X[mask, 2],
                    mode='markers',
                    marker=dict(size=3, color='rgba'+str(tuple((colors[i][:3]*255).astype(int))+ (0.85,))),
                    name=f'Cluster {label}'
                ))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                fig.add_trace(go.Scatter3d(
                    x=self.X[mask, 0],
                    y=self.X[mask, 1],
                    z=self.X[mask, 2],
                    mode='markers',
                    marker=dict(size=3, color='lightgray', opacity=0.6),
                    name='Noise'
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=self.X[mask, 0],
                    y=self.X[mask, 1],
                    z=self.X[mask, 2],
                    mode='markers',
                    marker=dict(size=3, color='rgba'+str(tuple((colors[i][:3]*255).astype(int))+ (0.85,))),
                    name=f'Cluster {label}'
                ))

        if centers is not None and len(centers) > 0:
            fig.add_trace(go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode='markers',
                marker=dict(size=5, color='black', symbol='x'),
                name='Centers'
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            legend=dict(
                itemsizing='constant'
            ),
            width=800,
            height=600,
        )
        if save_path is not None:
            fig.write_html(str(save_path))
            print(f"3D cluster plot saved to {save_path}")
        
        if show:
            fig.show()
        else:
            # close fig 
            plt.close()
            

    def run_kmeans(self, n_clusters: int = 10, random_state: int = 42, save_path: Optional[Path] = None) -> Dict:
        """
        Run K-Means clustering.
        Returns a dict with method, labels, centers, metrics, parameters, and model.
        """
        # For scikit-learn >= 1.4, n_init='auto' is OK; for older versions, use int
        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        except TypeError:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)

        labels = kmeans.fit_predict(self.X)
        centers = kmeans.cluster_centers_
        metrics = self._calculate_metrics(labels)

        if self.verbose:
            print(f"\nK-Means Results (k={n_clusters}):")
            print(f"  Clusters: {metrics['n_clusters']}")
            sil = metrics.get("silhouette_score")
            if sil is not None:
                print(f"  Silhouette Score: {sil:.3f}")
            self._plot_3d_clusters(labels, centers, f"K-Means Clustering (k={n_clusters})", save_path=save_path)

        return {
            "method": "kmeans",
            "labels": labels,
            "centers": centers,
            "metrics": metrics,
            "parameters": {"n_clusters": n_clusters, "random_state": random_state},
            "model": kmeans,
        }

    def run_dbscan(self, eps: float = 2.0, min_samples: int = 10, save_path: Optional[Path] = None) -> Dict:
        """
        Run DBSCAN clustering.
        Returns a dict with method, labels, centers (centroids), metrics, parameters, and model.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.X)

        # Centroids per cluster (exclude noise)
        centers = None
        unique_labels = set(labels) - {-1}
        if unique_labels:
            centers = np.array([self.X[labels == label].mean(axis=0) for label in sorted(unique_labels)])

        metrics = self._calculate_metrics(labels)

        if self.verbose:
            print(f"\nDBSCAN Results (eps={eps}, min_samples={min_samples}):")
            print(f"  Clusters: {metrics['n_clusters']}")
            print(f"  Noise points: {metrics['n_noise']} ({metrics['noise_ratio']:.1%})")
            sil = metrics.get("silhouette_score")
            if sil is not None:
                print(f"  Silhouette Score: {sil:.3f}")
            self._plot_3d_clusters(labels, centers, f"DBSCAN Clustering (eps={eps})", save_path=save_path)

        return {
            "method": "dbscan",
            "labels": labels,
            "centers": centers,
            "metrics": metrics,
            "parameters": {"eps": eps, "min_samples": min_samples},
            "model": dbscan,
        }

    def run_hdbscan(self, min_cluster_size: int = 10, min_samples: Optional[int] = None, save_path: Optional[Path] = None) -> Dict:
        """
        Run HDBSCAN clustering (if available).
        """
        if not HAVE_HDBSCAN:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(self.X)

        centers = None
        unique_labels = set(labels) - {-1}
        if unique_labels:
            centers = np.array([self.X[labels == label].mean(axis=0) for label in sorted(unique_labels)])

        metrics = self._calculate_metrics(labels)

        if self.verbose:
            print(f"\nHDBSCAN Results (min_cluster_size={min_cluster_size}, min_samples={min_samples}):")
            print(f"  Clusters: {metrics['n_clusters']}")
            print(f"  Noise points: {metrics['n_noise']} ({metrics['noise_ratio']:.1%})")
            sil = metrics.get("silhouette_score")
            if sil is not None:
                print(f"  Silhouette Score: {sil:.3f}")
            self._plot_3d_clusters(labels, centers, "HDBSCAN Clustering", save_path=save_path)

        return {
            "method": "hdbscan",
            "labels": labels,
            "centers": centers,
            "metrics": metrics,
            "parameters": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
            "model": clusterer,
        }

    # ---------- Optimization utilities ----------

    def find_optimal_k(self, k_range: Optional[range] = None, max_k: int = 80) -> Dict:
        """
        Find optimal number of clusters (K-Means) using multiple metrics.
        Returns the metrics over tested k, recommendations, and suggested_k.
        """
        n_samples = int(self.X.shape[0])

        # Determine feasible k values: 2 <= k <= min(max_k, n_samples - 1)
        k_max_feasible = max(2, min(int(max_k), max(2, n_samples - 1)))
        if k_range is None:
            k_range_low = list(range(2, int(0.5 * k_max_feasible) + 1, 1))
            k_range_high = list(range(int(0.5 * k_max_feasible) + 1, int(k_max_feasible) + 1, 3))
            k_values = k_range_low + k_range_high
        else:
            k_values = [k for k in list(k_range) if 2 <= int(k) <= k_max_feasible]

        # Ensure we have at least one candidate k
        if not k_values:
            # Fallback: if dataset too small, pick k=2 when possible
            if n_samples >= 3:
                k_values = [2]
            else:
                raise ValueError(
                    f"Not enough samples ({n_samples}) to perform K-Means optimization. "
                    "Need at least 3 samples."
                )

        if self.verbose:
            print(f"Optimizing K-Means clusters for k in {min(k_values)}-{max(k_values)} (bounded by n_samples-1={n_samples-1})...")

        results = {
            "k_values": k_values,
            "silhouette_scores": [],
            "calinski_harabasz_scores": [],
            "davies_bouldin_scores": [],
            "inertias": [],
        }

        for k in k_values:
            if self.verbose and k % 10 == 0:
                print(f"  Testing k={k}...")

            # KMeans fit
            try:
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            except TypeError:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

            labels = kmeans.fit_predict(self.X)
            results["inertias"].append(float(kmeans.inertia_))
            # Compute clustering metrics only when valid (need at least 2 clusters and < n_samples labels)
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)
            if 1 < n_labels < n_samples:
                try:
                    results["silhouette_scores"].append(float(silhouette_score(self.X, labels)))
                except Exception:
                    results["silhouette_scores"].append(float("nan"))
                try:
                    results["calinski_harabasz_scores"].append(float(calinski_harabasz_score(self.X, labels)))
                except Exception:
                    results["calinski_harabasz_scores"].append(float("nan"))
                try:
                    results["davies_bouldin_scores"].append(float(davies_bouldin_score(self.X, labels)))
                except Exception:
                    results["davies_bouldin_scores"].append(float("nan"))
            else:
                # Invalid for these metrics (e.g., k >= n_samples or degenerate clustering)
                results["silhouette_scores"].append(float("nan"))
                results["calinski_harabasz_scores"].append(float("nan"))
                results["davies_bouldin_scores"].append(float("nan"))

        # Determine best k per metric
        # Determine best k per metric, safely handling NaNs
        k_values_arr = np.array(k_values)
        sil_scores = np.array(results["silhouette_scores"], dtype=float)
        ch_scores = np.array(results["calinski_harabasz_scores"], dtype=float)
        db_scores = np.array(results["davies_bouldin_scores"], dtype=float)
        inertias = np.array(results["inertias"], dtype=float)

        def nan_argmax(a: np.ndarray) -> Optional[int]:
            if np.all(np.isnan(a)):
                return None
            return int(np.nanargmax(a))

        def nan_argmin(a: np.ndarray) -> Optional[int]:
            if np.all(np.isnan(a)):
                return None
            return int(np.nanargmin(a))

        best_silhouette_idx = nan_argmax(sil_scores)
        best_ch_idx = nan_argmax(ch_scores)
        best_db_idx = nan_argmin(db_scores)

        recommendations = {
            "silhouette": {
                "k": k_values[best_silhouette_idx] if best_silhouette_idx is not None else None,
                "score": float(sil_scores[best_silhouette_idx]) if best_silhouette_idx is not None else float("nan"),
            },
            "calinski_harabasz": {
                "k": k_values[best_ch_idx] if best_ch_idx is not None else None,
                "score": float(ch_scores[best_ch_idx]) if best_ch_idx is not None else float("nan"),
            },
            "davies_bouldin": {
                "k": k_values[best_db_idx] if best_db_idx is not None else None,
                "score": float(db_scores[best_db_idx]) if best_db_idx is not None else float("nan"),
            },
        }

        if self.verbose:
            self._plot_optimization_results(results)
            print("\nOptimization Results:")
            if recommendations["silhouette"]["k"] is not None:
                print(
                    f"  Best k by Silhouette: {recommendations['silhouette']['k']} "
                    f"(score: {recommendations['silhouette']['score']:.3f})"
                )
            if recommendations["calinski_harabasz"]["k"] is not None:
                print(
                    f"  Best k by Calinski-Harabasz: {recommendations['calinski_harabasz']['k']} "
                    f"(score: {recommendations['calinski_harabasz']['score']:.1f})"
                )
            if recommendations["davies_bouldin"]["k"] is not None:
                print(
                    f"  Best k by Davies-Bouldin: {recommendations['davies_bouldin']['k']} "
                    f"(score: {recommendations['davies_bouldin']['score']:.3f})"
                )

        # Choose suggested_k preference order: CH -> Silhouette -> DB -> Elbow (min inertia)
        suggested_k: Optional[int] = None
        if recommendations["calinski_harabasz"]["k"] is not None:
            suggested_k = int(recommendations["calinski_harabasz"]["k"]) 
        elif recommendations["silhouette"]["k"] is not None:
            suggested_k = int(recommendations["silhouette"]["k"]) 
        elif recommendations["davies_bouldin"]["k"] is not None:
            suggested_k = int(recommendations["davies_bouldin"]["k"]) 
        else:
            # Fallback: inertia elbow approximation -> pick min inertia k
            min_inertia_idx = int(np.argmin(inertias))
            suggested_k = int(k_values_arr[min_inertia_idx])

        return {
            "results": results,
            "recommendations": recommendations,
            "suggested_k": suggested_k,  # Prefer CH; fallback to other metrics or inertia
        }

    def estimate_natural_clusters(self, eps_range: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[float]]:
        """
        Estimate a natural number of clusters by scanning DBSCAN eps values.
        Returns (estimated_clusters, best_eps).
        """
        if eps_range is None:
            eps_range = np.arange(0.5, 10.0, 0.2)

        if self.verbose:
            print("\nEstimating natural clusters with DBSCAN:")

        best_eps: Optional[float] = None
        best_n_clusters = 0
        best_noise_ratio = 1.0

        for eps in eps_range:
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(self.X)
            n_clusters = len(set(labels) - {-1})
            noise_ratio = float(np.sum(labels == -1)) / float(len(labels))

            if self.verbose:
                print(f"  eps={eps:.1f}: {n_clusters:3d} clusters, {noise_ratio:.1%} noise")

            # Heuristic: reasonable clusters with low noise
            if 10 <= n_clusters <= 200 and noise_ratio < 0.15:
                if best_eps is None or (n_clusters > best_n_clusters and noise_ratio <= best_noise_ratio):
                    best_eps = float(eps)
                    best_n_clusters = int(n_clusters)
                    best_noise_ratio = float(noise_ratio)

        if best_eps is not None and self.verbose:
            print(f"  → Best: eps={best_eps}, {best_n_clusters} clusters, {best_noise_ratio:.1%} noise")

        return best_n_clusters if best_eps is not None else None, best_eps

    def _plot_optimization_results(self, results: Dict, save_path: Optional[Path] = None):
        """Plot optimization curves for different k-means metrics."""
        if not self.verbose:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(results["k_values"], results["silhouette_scores"], "bo-", markersize=4)
        axes[0, 0].set_xlabel("Number of clusters (k)")
        axes[0, 0].set_ylabel("Silhouette Score")
        axes[0, 0].set_title("Silhouette Analysis (Higher = Better)")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(results["k_values"], results["calinski_harabasz_scores"], "go-", markersize=4)
        axes[0, 1].set_xlabel("Number of clusters (k)")
        axes[0, 1].set_ylabel("Calinski-Harabasz Score")
        axes[0, 1].set_title("Calinski-Harabasz Index (Higher = Better)")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(results["k_values"], results["davies_bouldin_scores"], "ro-", markersize=4)
        axes[1, 0].set_xlabel("Number of clusters (k)")
        axes[1, 0].set_ylabel("Davies-Bouldin Score")
        axes[1, 0].set_title("Davies-Bouldin Index (Lower = Better)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(results["k_values"], results["inertias"], "mo-", markersize=4)
        axes[1, 1].set_xlabel("Number of clusters (k)")
        axes[1, 1].set_ylabel("Inertia")
        axes[1, 1].set_title("Elbow Method (Inertia)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Optimization plots saved to {save_path}")
        plt.show()

    def run_complete_analysis(self, max_k: int = 80, eps_range: Optional[np.ndarray] = None, save_path: Optional[Path] = None) -> Dict:
        """
        Run complete clustering analysis:
        1) Estimate natural clusters (DBSCAN)
        2) Optimize K for K-Means
        3) Run final K-Means and optional DBSCAN with best eps
        """
        if self.verbose:
            print("=" * 60)
            print("GALAXY CLUSTERING ANALYSIS")
            print("=" * 60)
            print(f"Dataset: {self.n_points} galaxies")

        # 1. Estimate DBSCAN eps and cluster count
        natural_k, best_eps = self.estimate_natural_clusters(eps_range=eps_range)

        # 2. Optimize K-Means
        target_max_k = max_k # min(max_k, (natural_k or 60) + 20)
        optimization = self.find_optimal_k(max_k=target_max_k)
        recommended_k = optimization["suggested_k"]

        if self.verbose:
            print(f"\n{'='*40}")
            print(f"FINAL ANALYSIS (k={recommended_k})")
            print(f"{'='*40}")

        # 3. Final runs
        final_results: Dict[str, object] = {
            "kmeans_optimized": self.run_kmeans(n_clusters=recommended_k, save_path=save_path),
            "natural_clustering": None,
            "optimization": optimization,
            "natural_estimate": {"clusters": natural_k, "eps": best_eps},
        }

        if best_eps is not None:
            final_results["natural_clustering"] = self.run_dbscan(eps=best_eps, min_samples=5, save_path=save_path)

        if self.verbose:
            print(f"\n{'='*60}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Recommended configuration: K-Means with k={recommended_k}")
            if natural_k:
                print(f"Natural clusters detected: {natural_k} (DBSCAN eps={best_eps})")

        return final_results


# ----------------------- Convenience functions -----------------------

def load_xyz(filepath: Union[str, Path]) -> np.ndarray:
    """Convenience function to load a 3D XYZ file into a NumPy array."""
    return GalaxyClusteringAnalyzer._load_data(Path(filepath))


