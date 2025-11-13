import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def compute_mean_density(positions, box_volume):
    """
    Compute the mean number density of galaxies.
    
    Parameters:
    -----------
    positions : np.ndarray
        Galaxy positions (N, 3)
    box_volume : float
        Total volume of the simulation box
        
    Returns:
    --------
    float : Mean number density (galaxies per unit volume)
    """
    return len(positions) / box_volume

def compute_density_contrast(n_galaxies, sphere_volume, rho_mean):
    """
    Compute density contrast δ = (ρ - ρ_mean) / ρ_mean
    
    Parameters:
    -----------
    n_galaxies : int
        Number of galaxies in the sphere
    sphere_volume : float
        Volume of the sphere
    rho_mean : float
        Mean density of galaxies
        
    Returns:
    --------
    float : Density contrast
    """
    rho = n_galaxies / sphere_volume
    return (rho - rho_mean) / rho_mean

def grow_sphere(center, all_galaxies, rho_mean, 
                initial_radius=5.0, radius_step=0.5, 
                min_radius=1.0, max_radius=100.0, 
                target_delta=-0.7, delta_tolerance=0.05):
    """
    Grow a sphere from a center point until density contrast reaches target.
    
    Parameters:
    -----------
    center : np.ndarray
        Center position (x, y, z)
    all_galaxies : np.ndarray
        All galaxy positions (N, 3)
    rho_mean : float
        Mean galaxy density
    initial_radius : float
        Starting radius for sphere growth
    radius_step : float
        Increment for radius at each step
    max_radius : float
        Maximum allowed radius
    target_delta : float
        Target density contrast (default: -0.7 for voids)
    delta_tolerance : float
        Tolerance for reaching target delta
    min_radius : float | None
        If provided, voids with final radius smaller than this will be considered invalid
        (converged=False) so callers can filter them out.
        
    Returns:
    --------
    dict : Dictionary containing void properties
    """
    # Build KD-tree for efficient radius queries
    tree = cKDTree(all_galaxies)
    
    radius = initial_radius
    void_properties = {
        'center': center,
        'radius': None,
        'n_galaxies': None,
        'density_contrast': None,
        'converged': False
    }
    
    while radius <= max_radius:
        # Find galaxies within current radius
        indices = tree.query_ball_point(center, radius)
        n_galaxies = len(indices)
        
        # Compute sphere volume
        sphere_volume = (4.0 / 3.0) * np.pi * radius**3
        
        # Compute density contrast
        if sphere_volume > 0:
            delta = compute_density_contrast(n_galaxies, sphere_volume, rho_mean)
        else:
            delta = 0.0
        
        # Check if we've reached the target density contrast
        if delta <= target_delta + delta_tolerance and delta >= target_delta - delta_tolerance:
            void_properties['radius'] = radius
            void_properties['n_galaxies'] = n_galaxies
            void_properties['density_contrast'] = delta
            void_properties['converged'] = True
            break
        
        # If density contrast is too low (void is too empty), stop
        if delta < target_delta - delta_tolerance:
            void_properties['radius'] = radius - radius_step  # Use previous radius
            prev_indices = tree.query_ball_point(center, radius - radius_step)
            prev_volume = (4.0 / 3.0) * np.pi * (radius - radius_step)**3
            void_properties['n_galaxies'] = len(prev_indices)
            void_properties['density_contrast'] = compute_density_contrast(
                len(prev_indices), prev_volume, rho_mean
            )
            void_properties['converged'] = True
            break
        
        # Increment radius
        radius += radius_step
    
    # If max radius reached without convergence
    if not void_properties['converged']:
        void_properties['radius'] = max_radius
        indices = tree.query_ball_point(center, max_radius)
        sphere_volume = (4.0 / 3.0) * np.pi * max_radius**3
        void_properties['n_galaxies'] = len(indices)
        void_properties['density_contrast'] = compute_density_contrast(
            len(indices), sphere_volume, rho_mean
        )
    
    # Apply minimum radius criterion
    if void_properties['radius'] is not None:
        if void_properties['radius'] < float(min_radius):
            # mark as non-converged/invalid so the caller can drop it
            void_properties['converged'] = False
    
    return void_properties

def find_all_voids(cluster_centers, all_galaxies, box_dimensions,
                   initial_radius=1.0, radius_step=0.5, 
                   min_radius=2.0, max_radius=100.0, 
                   target_delta=-0.7, delta_tolerance=0.05,):
    """
    Find voids around all cluster centers (CORRECTED VERSION).
    
    Spheres grow from void centers (δ << -0.7) outward until δ reaches -0.7.
    Density contrast INCREASES (becomes less negative) as radius increases.
    
    Parameters:
    -----------
    cluster_centers : np.ndarray
        Cluster center positions (M, 3)
    all_galaxies : np.ndarray
        All galaxy positions (N, 3)
    box_dimensions : tuple or np.ndarray
        Box dimensions (Lx, Ly, Lz) for volume calculation
    initial_radius : float
        Starting radius for sphere growth (should be small, e.g., 1.0 Mpc/h)
    radius_step : float
        Increment for radius at each step
    max_radius : float
        Maximum allowed radius
    target_delta : float
        Target density contrast (default: -0.7)
    delta_tolerance : float
        Tolerance for reaching target delta
    min_radius : float | None
        If provided, voids with final radius smaller than this will be DROPPED from the result.
        
    Returns:
    --------
    list : List of void properties dictionaries
    """
    # Compute box volume and mean density
    box_volume = np.prod(box_dimensions)
    rho_mean = compute_mean_density(all_galaxies, box_volume)
    
    # Build KD-tree for efficient queries
    tree = cKDTree(all_galaxies)
    
    print(f"Mean galaxy density: {rho_mean:.6f} galaxies per unit volume^3")
    print(f"Total galaxies: {len(all_galaxies)}")
    print(f"Box volume: {box_volume:.2f}")
    print(f"Growing {len(cluster_centers)} voids...")
    print(f"Target δ = {target_delta} ± {delta_tolerance}\n")
    
    voids = []
    
    for i, center in enumerate(cluster_centers):
        void = {
            'center': center,
            'void_id': i,
            'converged': False
        }
        
        radius = initial_radius
        prev_radius = initial_radius
        prev_delta = None
        prev_n_galaxies = 0
        
        # Grow sphere from center outward
        while radius <= max_radius:
            # Find galaxies within current radius
            indices = tree.query_ball_point(center, radius)
            n_galaxies = len(indices)
            
            # Compute sphere volume and density contrast
            sphere_volume = (4.0 / 3.0) * np.pi * radius**3
            
            if sphere_volume > 0:
                delta = compute_density_contrast(n_galaxies, sphere_volume, rho_mean)
            else:
                delta = 0.0
            
            # Check convergence: δ should INCREASE toward -0.7
            if target_delta - delta_tolerance <= delta <= target_delta + delta_tolerance:
                # Within tolerance - converged!
                void['radius'] = radius
                void['n_galaxies'] = n_galaxies
                void['density_contrast'] = delta
                void['converged'] = True
                break
                
            elif delta > target_delta + delta_tolerance:
                # Overshot the target (δ > -0.7, too dense)
                # Use previous radius which was closer to target
                if prev_delta is not None:
                    # Choose the radius with δ closer to target
                    if abs(prev_delta - target_delta) < abs(delta - target_delta):
                        void['radius'] = prev_radius
                        void['n_galaxies'] = prev_n_galaxies
                        void['density_contrast'] = prev_delta
                    else:
                        void['radius'] = radius
                        void['n_galaxies'] = n_galaxies
                        void['density_contrast'] = delta
                else:
                    # No previous step, use current
                    void['radius'] = radius
                    void['n_galaxies'] = n_galaxies
                    void['density_contrast'] = delta
                
                void['converged'] = True
                break
            
            # Store current values as previous for next iteration
            prev_radius = radius
            prev_delta = delta
            prev_n_galaxies = n_galaxies
            
            # Increment radius
            radius += radius_step
        
        # If max radius reached without convergence
        if not void['converged']:
            void['radius'] = max_radius
            indices = tree.query_ball_point(center, max_radius)
            sphere_volume = (4.0 / 3.0) * np.pi * max_radius**3
            void['n_galaxies'] = len(indices)
            void['density_contrast'] = compute_density_contrast(
                len(indices), sphere_volume, rho_mean
            )
            print(f"Warning: Void {i} reached max radius with δ = {void['density_contrast']:.3f}")
        
        # Apply minimum radius filter if requested
        if min_radius is None or ('radius' in void and void['radius'] >= float(min_radius)):
            voids.append(void)
        else:
            # Skip adding too-small voids
            pass
        
        if (i + 1) % 10 == 0 or (i + 1) == len(cluster_centers):
            print(f"Processed {i + 1}/{len(cluster_centers)} voids")
    
    # Filter converged voids
    converged_voids = [v for v in voids if v['converged']]
    print(f"\nConverged voids: {len(converged_voids)}/{len(voids)}")
    
    return voids

def filter_overlapping_voids(voids, overlap_threshold=0.5):
    """
    Remove overlapping voids, keeping the one with density contrast closest to target.
    
    Parameters:
    -----------
    voids : list
        List of void dictionaries
    overlap_threshold : float
        Fraction of overlap to consider voids as overlapping
        
    Returns:
    --------
    list : Filtered list of non-overlapping voids
    """
    filtered_voids = []
    
    for void in sorted(voids, key=lambda v: abs(v['density_contrast'] - (-0.7))):
        is_overlapping = False
        
        for existing_void in filtered_voids:
            # Compute distance between centers
            dist = np.linalg.norm(void['center'] - existing_void['center'])
            sum_radii = void['radius'] + existing_void['radius']
            
            # Check for overlap
            if dist < overlap_threshold * sum_radii:
                is_overlapping = True
                break
        
        if not is_overlapping:
            filtered_voids.append(void)
    
    return filtered_voids

def grow_all_spheres_simultaneously(cluster_centers, all_galaxies, rho_mean,
                                             initial_radius=1.0, radius_step=0.5,
                                             min_radius=2.0, max_radius=100.0,
                                             target_delta=-0.7, delta_tolerance=0.05):
    """
    Grow all spheres simultaneously from void centers.
    Spheres start very underdense (δ << -0.7) and grow until δ reaches -0.7.
    If min_radius is provided, voids with final radius < min_radius are removed from results.
    """
    tree = cKDTree(all_galaxies)
    n_voids = len(cluster_centers)
    
    voids = [{
        'center': center,
        'radius': initial_radius,
        'converged': False,
        'void_id': i,
        'n_galaxies': 0,
        'density_contrast': -1.0
    } for i, center in enumerate(cluster_centers)]
    
    print(f"Growing {n_voids} voids simultaneously...")
    print(f"Starting radius: {initial_radius}, Step: {radius_step}, Max: {max_radius}")
    print(f"Target δ = {target_delta} ± {delta_tolerance}\n")
    
    step_count = 0
    max_steps = int((max_radius - initial_radius) / radius_step) + 1
    
    while step_count < max_steps:
        active_voids = [v for v in voids if not v['converged']]
        
        if len(active_voids) == 0:
            print("All voids converged!")
            break
        
        # Grow all active voids by one step
        for void in active_voids:
            current_radius = void['radius']
            
            # Find galaxies within current radius
            indices = tree.query_ball_point(void['center'], current_radius)
            n_galaxies = len(indices)
            
            # Compute sphere volume and density contrast
            sphere_volume = (4.0 / 3.0) * np.pi * current_radius**3
            delta = compute_density_contrast(n_galaxies, sphere_volume, rho_mean)
            
            # Update void properties
            void['n_galaxies'] = n_galaxies
            void['density_contrast'] = delta
            
            # Check convergence: δ should be INCREASING toward -0.7
            # We converge when δ is in the range [target - tol, target + tol]
            if target_delta - delta_tolerance <= delta <= target_delta + delta_tolerance:
                # Within tolerance - converged!
                void['converged'] = True
                
            elif delta > target_delta + delta_tolerance:
                # We've gone past the target (δ > -0.7, too dense)
                # This means the void boundary was at the previous radius
                # Revert to previous radius
                if current_radius > initial_radius + radius_step:
                    void['radius'] = current_radius - radius_step
                    prev_indices = tree.query_ball_point(void['center'], void['radius'])
                    prev_vol = (4.0/3.0) * np.pi * void['radius']**3
                    void['n_galaxies'] = len(prev_indices)
                    void['density_contrast'] = compute_density_contrast(
                        len(prev_indices), prev_vol, rho_mean
                    )
                void['converged'] = True
            
            # If still active, increment radius for next iteration
            if not void['converged']:
                void['radius'] += radius_step
        
        step_count += 1
        
        if step_count % 20 == 0:
            converged_count = sum(1 for v in voids if v['converged'])
            active = [v for v in voids if not v['converged']]
            if active:
                mean_radius = np.mean([v['radius'] for v in active])
                mean_delta = np.mean([v['density_contrast'] for v in active])
                min_delta = np.min([v['density_contrast'] for v in active])
                max_delta = np.max([v['density_contrast'] for v in active])
                print(f"Step {step_count}: Active voids mean R = {mean_radius:.1f} Mpc/h, "
                      f"δ range = [{min_delta:.3f}, {max_delta:.3f}], "
                      f"Converged = {converged_count}/{n_voids}")
    
    # Handle non-converged voids (reached max_radius)
    for void in voids:
        if not void['converged']:
            void['radius'] = max_radius
            indices = tree.query_ball_point(void['center'], max_radius)
            sphere_volume = (4.0 / 3.0) * np.pi * max_radius**3
            void['n_galaxies'] = len(indices)
            void['density_contrast'] = compute_density_contrast(
                len(indices), sphere_volume, rho_mean
            )
            print(f"Warning: Void {void['void_id']} reached max radius with δ = {void['density_contrast']:.3f}")
    
    converged_count = sum(1 for v in voids if v['converged'])
    print(f"\nFinal: {converged_count}/{n_voids} voids converged")

    # Filter out too-small voids if requested
    if min_radius is not None:
        before = len(voids)
        voids = [v for v in voids if ('radius' in v and v['radius'] >= float(min_radius))]
        after = len(voids)
        if before != after:
            print(f"Applied min_radius={min_radius:.3f}: kept {after}/{before} voids")
    
    return voids

def diagnose_void_growth(center, all_galaxies, rho_mean, max_radius=20.0, step=0.1):
    """
    Diagnose the density contrast evolution for a single void center.
    Useful for understanding why voids converge quickly or not at all.
    """
    tree = cKDTree(all_galaxies)
    
    radii = []
    deltas = []
    n_gals = []
    
    r = step
    while r <= max_radius:
        indices = tree.query_ball_point(center, r)
        n = len(indices)
        vol = (4.0 / 3.0) * np.pi * r**3
        delta = compute_density_contrast(n, vol, rho_mean)
        
        radii.append(r)
        deltas.append(delta)
        n_gals.append(n)
        
        r += step
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(radii, deltas, 'b-', linewidth=2)
    ax1.axhline(y=-0.3, color='r', linestyle='--', label='Target δ = -0.3')
    ax1.axhline(y=-0.35, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(y=-0.25, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.fill_between(radii, -0.35, -0.25, alpha=0.2, color='orange', label='Tolerance range')
    ax1.set_xlabel('Radius [Mpc/h]', fontsize=12)
    ax1.set_ylabel('Density Contrast δ', fontsize=12)
    ax1.set_title('Density Contrast vs Radius', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, max_radius)
    ax1.set_ylim(-1, 5)
    
    ax2.plot(radii, n_gals, 'g-', linewidth=2)
    ax2.set_xlabel('Radius [Mpc/h]', fontsize=12)
    ax2.set_ylabel('Number of Galaxies', fontsize=12)
    ax2.set_title('Galaxy Count vs Radius', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return radii, deltas, n_gals