from __future__ import annotations

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from typing import Optional, Tuple


def _make_kgrid(N: int, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D wave-number grid (radians per length) for a periodic box [-L/2, L/2]^3 with N cells per side.
    Returns (kx, ky, kz, k2).
    """
    k1d = 2.0 * np.pi * fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def _bbks_transfer(k: np.ndarray, Om: float, h: float) -> np.ndarray:
    """
    BBKS transfer function (Bardeen et al. 1986), using simple Gamma = Om*h.
    Good enough for synthetic fields without baryon wiggles.

    T(q) = ln(1+2.34q)/(2.34q) * [1 + 3.89q + (16.1q)^2 + (5.46q)^3 + (6.71q)^4]^{-1/4}
    with q = k / Gamma.
    """
    Gamma = Om * h
    q = np.maximum(k / np.maximum(Gamma, 1e-8), 1e-12)
    L0 = np.log(1.0 + 2.34 * q) / (2.34 * q)
    C0 = (1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25)
    return L0 * C0


def _gaussian_field_from_power(
    N: int,
    L: float,
    Om: float = 0.315,
    h: float = 0.674,
    ns: float = 0.965,
    R_smooth: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a zero-mean Gaussian random field with CDM-like power spectrum on an N^3 grid.

    - P(k) ∝ k^ns T^2(k) (BBKS) with optional Gaussian smoothing exp(-k^2 R_smooth^2/2)
    - Uses 'filtering white noise in Fourier space' approach; overall amplitude is scaled to unit std.

    Returns:
        delta_G (N,N,N) float32, mean ~ 0, std ~ 1 (before lognormal mapping).
    """
    rng = rng or np.random.default_rng()
    # Real-space white noise
    w = rng.normal(0.0, 1.0, size=(N, N, N)).astype(np.float32)

    # Fourier filter sqrt(P(k)) (shape only; overall amplitude set by rescaling in real space)
    kx, ky, kz, k2 = _make_kgrid(N, L)
    k = np.sqrt(k2)
    T = _bbks_transfer(k, Om, h)
    P_shape = (np.power(np.maximum(k, 1e-8), ns) * (T ** 2)).astype(np.float64)

    # Optional smoothing to control filament thickness
    if R_smooth > 0.0:
        P_shape *= np.exp(-0.5 * (k * R_smooth) ** 2)

    F = np.sqrt(P_shape, dtype=np.float64)
    F[0, 0, 0] = 0.0  # no DC power

    Wk = fftn(w, norm=None)
    field = ifftn(Wk * F, norm=None).real.astype(np.float32)

    # Normalize to unit variance (shape-only field)
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def _lognormal_intensity_from_gaussian(
    gfield: np.ndarray,
    nbar: float,
    bias: float = 1.5,
) -> np.ndarray:
    """
    Create a lognormal intensity field (galaxies per volume) from a zero-mean Gaussian field.

    We use λ(x) = nbar * exp(b*g - (b^2 * σ_g^2)/2), ensuring ⟨λ⟩ = nbar.
    """
    sigma2 = float(np.var(gfield))
    mu_shift = -0.5 * (bias ** 2) * sigma2
    lam = nbar * np.exp(bias * gfield + mu_shift)
    return lam.astype(np.float32)


def _poisson_sample_from_cell_intensity(
    lam: np.ndarray,
    L: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a Poisson point process with cell-wise intensities λ_ijk (per unit volume).
    - lam is shape (N,N,N) with mean nbar
    - Returns galaxy positions in [-L/2, L/2]^3 with periodic boundary conditions.
    """
    rng = rng or np.random.default_rng()
    N = lam.shape[0]
    assert lam.shape == (N, N, N)

    cell_vol = (L / N) ** 3
    # Number of points per cell
    Ni = rng.poisson(lam * cell_vol)

    total = int(Ni.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Flatten indices of occupied cells
    idx_nonzero = np.flatnonzero(Ni)
    counts = Ni.ravel()[idx_nonzero]
    # Expand cell indices
    # Convert flat idx to 3D indices
    iz = idx_nonzero % N
    iy = (idx_nonzero // N) % N
    ix = idx_nonzero // (N * N)

    # Repeat each cell index according to its count
    rep = np.repeat(np.arange(idx_nonzero.size), counts)
    cx = ix[rep]
    cy = iy[rep]
    cz = iz[rep]

    # Sample uniform offsets inside cell
    u = rng.random((total, 3), dtype=np.float32)
    dx = L / N

    # Positions in [0, L)
    pos = (np.column_stack([cx, cy, cz]).astype(np.float32) + u) * dx
    # Shift to [-L/2, L/2)
    pos -= L / 2.0
    return pos.astype(np.float32)


def _zeldovich_displacement(
    gaussian_field: np.ndarray,
    L: float,
    D: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Zel'dovich displacement field ψ(x) from Gaussian overdensity δ(x):
    ψ_k = i k / k^2 δ_k (up to a proportionality constant absorbed into D).

    Returns displacement components (psi_x, psi_y, psi_z), each (N,N,N) float32.
    """
    N = gaussian_field.shape[0]
    delta_k = fftn(gaussian_field.astype(np.float32), norm=None)

    kx, ky, kz, k2 = _make_kgrid(N, L)
    k2_safe = k2.copy()
    k2_safe[0, 0, 0] = 1.0  # avoid division by zero

    factor = (1j / k2_safe).astype(np.complex64)
    psi_x_k = factor * (kx.astype(np.float32) * delta_k.astype(np.complex64))
    psi_y_k = factor * (ky.astype(np.float32) * delta_k.astype(np.complex64))
    psi_z_k = factor * (kz.astype(np.float32) * delta_k.astype(np.complex64))

    psi_x = ifftn(psi_x_k, norm=None).real.astype(np.float32) * D
    psi_y = ifftn(psi_y_k, norm=None).real.astype(np.float32) * D
    psi_z = ifftn(psi_z_k, norm=None).real.astype(np.float32) * D
    return psi_x, psi_y, psi_z


def _tricubic_periodic_sample(vec_field: Tuple[np.ndarray, np.ndarray, np.ndarray],  # (Fx,Fy,Fz)
                              points: np.ndarray,
                              L: float) -> np.ndarray:
    """
    Trilinear interpolation of a vector field on a periodic grid at arbitrary points in [-L/2,L/2)^3.
    Returns array of shape (Npts, 3).
    """
    Fx, Fy, Fz = vec_field
    N = Fx.shape[0]
    dx = L / N

    # Map positions from [-L/2, L/2) to [0, N)
    q = (points + L / 2.0) / dx
    q = np.mod(q, N)  # periodic wrap
    i0 = np.floor(q).astype(np.int64)
    t = (q - i0).astype(np.float32)

    # Corner indices
    i1 = (i0 + 1) % N

    def gather(F):
        # 8 corners
        c000 = F[i0[:, 0], i0[:, 1], i0[:, 2]]
        c100 = F[i1[:, 0], i0[:, 1], i0[:, 2]]
        c010 = F[i0[:, 0], i1[:, 1], i0[:, 2]]
        c110 = F[i1[:, 0], i1[:, 1], i0[:, 2]]
        c001 = F[i0[:, 0], i0[:, 1], i1[:, 2]]
        c101 = F[i1[:, 0], i0[:, 1], i1[:, 2]]
        c011 = F[i0[:, 0], i1[:, 1], i1[:, 2]]
        c111 = F[i1[:, 0], i1[:, 1], i1[:, 2]]

        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        c = c0 * (1 - tz) + c1 * tz
        return c.astype(np.float32)

    disp_x = gather(Fx)
    disp_y = gather(Fy)
    disp_z = gather(Fz)
    return np.stack([disp_x, disp_y, disp_z], axis=1)


def sample_lognormal_zeldovich_background(
    box_size_mpc: float,
    target_number_density: float,
    ngrid: int = 128,
    R_smooth: float = 1.5,
    bias: float = 1.6,
    flow_strength: float = 3.0,
    Om: float = 0.315,
    h: float = 0.674,
    ns: float = 0.965,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate background galaxies that look like a cosmic web:
      1) Gaussian field with CDM-like power (BBKS) -> smoothed
      2) Lognormal mapping to intensity λ(x) with bias
      3) Poisson sampling per cell
      4) Zel'dovich displacements to produce filaments/nodes

    Args:
        box_size_mpc: L (Mpc/h), periodic cube [-L/2, L/2]^3
        target_number_density: ⟨n⟩ in galaxies / (Mpc/h)^3
        ngrid: grid resolution for fields and displacements
        R_smooth: Gaussian smoothing (Mpc/h). Larger = thicker filaments, fewer small clumps.
        bias: linear bias for tracers in the lognormal mapping (1.2–2 typical)
        flow_strength: overall amplitude for Zel'dovich displacements (try 2–8)
        Om,h,ns: cosmological-shape parameters (only set the large-scale shape)

    Returns:
        positions: (Ngal,3) float32 within [-L/2,L/2], periodic.
    """
    rng = rng or np.random.default_rng()

    # 1) Gaussian field with desired shape
    gfield = _gaussian_field_from_power(
        N=ngrid, L=box_size_mpc, Om=Om, h=h, ns=ns, R_smooth=R_smooth, rng=rng
    )

    # 2) Lognormal intensity field with given mean density and bias
    lam = _lognormal_intensity_from_gaussian(
        gfield=gfield, nbar=target_number_density, bias=bias
    )

    # 3) Poisson-sample galaxies per cell
    positions = _poisson_sample_from_cell_intensity(lam, L=box_size_mpc, rng=rng)

    if positions.shape[0] == 0:
        return positions

    # 4) Zel'dovich displacement field from the same Gaussian field (before lognormal)
    psi = _zeldovich_displacement(gfield, L=box_size_mpc, D=flow_strength)
    disp = _tricubic_periodic_sample(psi, positions, L=box_size_mpc)

    positions = positions + disp
    # Periodic wrap back to box
    L = box_size_mpc
    positions = (positions + L / 2.0) % L - L / 2.0
    return positions.astype(np.float32)