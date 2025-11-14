import numpy as np

from scipy.spatial import cKDTree
from typing import Optional, Tuple
from scipy.spatial import Voronoi
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi, ConvexHull
from collections import defaultdict


class GalaxyDataPreparer:
    def __init__(self):
        pass


def voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    Returns regions and vertices (adapted from scipy cookbook examples).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0: v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                # Both vertices are finite, no need to add a new one
                continue

            # Compute the missing endpoint at infinity
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Sort region vertices counterclockwise
        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def compute_voronoi_slice(z0: float, z_range: Tuple[float, float], unique_points: np.ndarray, slab_frac: float = 0.02) -> Tuple[np.ndarray, list]:
    """Compute 2D Voronoi on galaxies within a thin slab around z0.
    Returns (vertices, regions) for plotting. Slab thickness is slab_frac * (z_max - z_min).
    """
    zmin, zmax = z_range
    slab_th = slab_frac * (zmax - zmin)
    mask = np.abs(unique_points[:, 2] - z0) <= (slab_th / 2.0)
    pts2 = unique_points[mask][:, :2]
    if pts2.shape[0] < 3:
        print(f"Slice at z={z0:.3f}: not enough points ({pts2.shape[0]})")
        return np.empty((0, 2)), []
    vor2 = Voronoi(pts2)
    regions, vertices = voronoi_finite_polygons_2d(vor2)
    return vertices, regions

def plot_voronoi_slice(z0: float, z_range: Tuple[float, float], unique_points: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, slab_frac: float = 0.02, ax=None, cmap="viridis"):
    verts, regions = compute_voronoi_slice(z0, z_range=z_range, unique_points=unique_points, slab_frac=slab_frac)
    if verts.size == 0 or len(regions) == 0:
        return None
    polys = []
    areas = []
    for reg in regions:
        poly = verts[np.asarray(reg)]
        # Optional: clip to bbox for nicer visuals (simple coordinate clamping)
        poly[:, 0] = np.clip(poly[:, 0], x_min, x_max)
        poly[:, 1] = np.clip(poly[:, 1], y_min, y_max)
        polys.append(poly)
        # polygon area for simple coloring (shoelace)
        x, y = poly[:, 0], poly[:, 1]
        a = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(a)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    coll = PolyCollection(polys, array=np.asarray(areas), cmap=cmap, edgecolor='k', linewidths=0.2, alpha=0.6)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Voronoi slice near z={z0:.3f}")
    plt.colorbar(coll, ax=ax, label="cell area (2D)")
    return ax


#########################################################################################################################
################################ Minimal Voronoi features: coords, volumes, adjacency ##################################
#########################################################################################################################

def polyhedron_volume(points3d: np.ndarray) -> float:
    """
    Compute the volume of a convex polyhedron from its vertices via tetrahedral
    decomposition using the ConvexHull surface and a reference point at the
    mean of hull vertices. Returns 0.0 for degenerate inputs (<4 points).
    """
    if points3d.shape[0] < 4:
        return 0.0
    hull = ConvexHull(points3d)
    O = points3d[hull.vertices].mean(axis=0)
    vol_total = 0.0
    for tri in hull.simplices:
        A, B, C = points3d[tri]
        tetra_vol = abs(np.dot(np.cross(B - O, C - O), A - O)) / 6.0
        vol_total += tetra_vol
    return vol_total


def build_voronoi_graph_and_volumes(
    vor: Voronoi,
    clip_bbox=None,
) -> dict:
    """
    Build minimal features from a 3D SciPy Voronoi object:
      - points: (N,3) input site coordinates
      - volumes: (N,) finite cell volumes (nan for unbounded or clipped cells)
      - indptr, indices: CSR adjacency of the ridge graph (finite ridges only)

    clip_bbox: optional (xmin, xmax, ymin, ymax, zmin, zmax). If provided,
               any cell with a vertex outside the bbox is treated as unbounded.
    """
    points = vor.points
    N = points.shape[0]

    volumes = np.full(N, np.nan, dtype=float)
    finite_mask = np.zeros(N, dtype=bool)

    regions = vor.regions
    point_region = vor.point_region
    vertices = vor.vertices

    # Per-cell volume for finite regions
    for i in range(N):
        region_id = point_region[i]
        region = regions[region_id]
        if len(region) == 0 or (-1 in region):
            continue  # unbounded
        poly_vertices = vertices[region]
        if clip_bbox is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = clip_bbox
            inside = (
                (poly_vertices[:, 0] >= xmin) & (poly_vertices[:, 0] <= xmax) &
                (poly_vertices[:, 1] >= ymin) & (poly_vertices[:, 1] <= ymax) &
                (poly_vertices[:, 2] >= zmin) & (poly_vertices[:, 2] <= zmax)
            )
            if not np.all(inside):
                continue
        volumes[i] = polyhedron_volume(poly_vertices)
        finite_mask[i] = True

    # Neighbor graph via finite ridges
    neighbor_sets = defaultdict(set)
    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        p, q = ridge_points
        if (-1 in ridge_vertices) or len(ridge_vertices) == 0:
            continue  # infinite ridge
        neighbor_sets[p].add(q)
        neighbor_sets[q].add(p)

    indptr = [0]
    indices = []
    for i in range(N):
        nbrs = sorted(neighbor_sets[i])
        indices.extend(nbrs)
        indptr.append(len(indices))
    indptr = np.array(indptr, dtype=np.int64)
    indices = np.array(indices, dtype=np.int32)

    return {
        "points": points,
        "volumes": volumes,
        "finite_mask": finite_mask,
        "indptr": indptr,
        "indices": indices,
    }


## Removed legacy build_voronoi_features_from_vor in favor of build_voronoi_graph_and_volumes


# -----------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------

def save_features_npz(features: dict, path: str):
    """
    Save feature dict to a compressed NPZ.
    Skips None entries.
    """
    clean = {k: v for k, v in features.items() if v is not None}
    np.savez_compressed(path, **clean)

def load_features_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# -----------------------------------------------------------
# Graph ML helper
# -----------------------------------------------------------

def build_edge_index_from_csr(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Convert CSR adjacency (indptr, indices) into edge_index (2,E) with
    directed edges (i -> neighbor).
    """
    N = indptr.size - 1
    counts = np.diff(indptr)
    sources = np.repeat(np.arange(N, dtype=np.int64), counts)
    edge_index = np.vstack([sources, indices.astype(np.int64)])
    return edge_index


## Removed higher-level feature matrix assembly per new minimal spec


# -----------------------------------------------------------
# Example usage (guarded)
# -----------------------------------------------------------

if __name__ == "__main__":
    # Demonstration with random points
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, size=(200, 3))
    vor = Voronoi(pts)

    features = build_voronoi_graph_and_volumes(
        vor,
        clip_bbox=None,
    )

    save_features_npz(features, "demo_voronoi_features_min.npz")

    edge_index = build_edge_index_from_csr(features["indptr"], features["indices"])
    np.save("demo_edge_index.npy", edge_index)

    print("Saved minimal demo features and edge_index.")