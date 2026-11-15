"""Static fault geometry and deterministic spatial coarsening.

Triangle connectivity is preferred. radius_knn is an explicitly approximate
fallback for datasets that only provide element centres. All arrays preserve
the original fault-column numbering.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from scipy.spatial import cKDTree

GEOMETRY_VERSION = "fault_geometry_v1"


@dataclass
class FaultGeometry:
    fine_xyz_km: np.ndarray
    coarse_xyz_km: np.ndarray
    fine_area: np.ndarray
    coarse_area: np.ndarray
    edge_index: np.ndarray
    pairs: np.ndarray
    parent: np.ndarray
    slot: np.ndarray
    metadata: dict

    def validate(self):
        n = len(self.fine_xyz_km)
        if n < 1 or self.fine_xyz_km.shape != (n, 3):
            raise ValueError("Expected nonempty [N,3] element coordinates")
        if (
            not np.isfinite(self.fine_xyz_km).all()
            or not np.isfinite(self.coarse_xyz_km).all()
        ):
            raise ValueError("Non-finite coordinates")
        if self.fine_area.shape != (n,) or np.any(self.fine_area <= 0):
            raise ValueError("Element areas must be positive")
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [coarse_tokens,2]")
        if np.any(self.pairs < -1) or np.any(self.pairs >= n):
            raise ValueError("Invalid child index")
        valid = self.pairs[self.pairs >= 0]
        if not np.array_equal(np.sort(valid), np.arange(n)):
            raise ValueError("Every real element must appear exactly once")
        if np.any(self.pairs[:, 0] < 0):
            raise ValueError("Empty parent token")
        if self.parent.shape != (n,) or self.slot.shape != (n,):
            raise ValueError("Invalid inverse mapping dimensions")
        if not np.array_equal(self.pairs[self.parent, self.slot], np.arange(n)):
            raise ValueError("Parent/child mapping is not invertible")
        m = len(self.pairs)
        if self.coarse_xyz_km.shape != (m, 3) or self.coarse_area.shape != (m,):
            raise ValueError("Invalid coarse geometry dimensions")
        if not np.isclose(self.coarse_area.sum(), self.fine_area.sum()):
            raise ValueError("Coarsening must conserve covered area")
        edges = {tuple(e) for e in self.edge_index.tolist()}
        for a, b in self.pairs:
            if b >= 0 and tuple(sorted((int(a), int(b)))) not in edges:
                raise ValueError("A parent joins non-neighbouring elements")
        return self


def project_llh(llh, reference_lonlat=None):
    """Spherical azimuthal-equidistant horizontal coordinates and positive depth.

    llh columns: longitude (degrees), latitude (degrees), negative depth (km).
    Circular longitude differences also support grids crossing the date line.
    """
    llh = np.asarray(llh, dtype=np.float64)
    if llh.ndim != 2 or llh.shape[1] != 3 or not np.isfinite(llh).all():
        raise ValueError("llh must be a finite [N,3] array")
    if np.any(np.abs(llh[:, 1]) > 90) or np.any(llh[:, 2] > 1e-6):
        raise ValueError("Expected latitude in degrees and non-positive depth in km")
    lon, lat = np.deg2rad(llh[:, :2]).T
    lon0 = np.arctan2(np.sin(lon).mean(), np.cos(lon).mean())
    lat0 = lat.mean()
    if reference_lonlat is not None:
        ref = np.asarray(reference_lonlat, dtype=float)
        if ref.shape != (2,) or not np.isfinite(ref).all() or abs(ref[1]) > 90:
            raise ValueError("Invalid fixed projection reference")
        lon0, lat0 = np.deg2rad(ref)
    dl = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    cc = np.clip(
        np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dl), -1, 1
    )
    c = np.arccos(cc)
    if np.any(c > np.pi / 2):
        raise ValueError("Use a regional grid covering less than a hemisphere")
    k = np.ones_like(c)
    np.divide(c, np.sin(c), out=k, where=c > 1e-10)
    east = 6371.0 * k * np.cos(lat) * np.sin(dl)
    north = (
        6371.0
        * k
        * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dl))
    )
    return np.c_[east, north, -llh[:, 2]]


def _canonical_edges(edges, n):
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Adjacency must have shape [E,2], with zero-based element IDs")
    if not np.equal(edges, np.floor(edges)).all():
        raise ValueError("Adjacency indices must be integers")
    edges = edges.astype(np.int64)
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Adjacency indices do not match the fault-element count")
    edges = np.sort(edges, axis=1)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def triangle_edges(triangles):
    """Triangle row IDs must correspond exactly to input field columns."""
    triangles = np.asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangle connectivity must be [N,3] vertex IDs")
    if not np.equal(triangles, np.floor(triangles)).all() or np.any(triangles < 0):
        raise ValueError("Triangle vertex IDs must be nonnegative integers")
    triangles = triangles.astype(np.int64)
    owner, edges = {}, []
    for i, tri in enumerate(triangles):
        if len(set(tri.tolist())) != 3:
            raise ValueError("Degenerate triangle")
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            owners = owner.setdefault(key, [])
            if len(owners) == 2:
                raise ValueError("Non-manifold triangle edge has more than two owners")
            for j in owners:
                edges.append((j, i))
            owners.append(i)
    return _canonical_edges(edges, len(triangles))


def radius_knn_edges(xyz, k=6, max_distance_km=30.0):
    n = len(xyz)
    if int(k) < 1 or float(max_distance_km) <= 0:
        raise ValueError("knn k and maximum distance must be positive")
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    distance, index = cKDTree(xyz).query(xyz, k=min(int(k) + 1, n))
    edges = []
    for i in range(n):
        for d, j in zip(distance[i, 1:], index[i, 1:]):
            if np.isfinite(d) and d <= max_distance_km:
                edges.append((i, int(j)))
    return _canonical_edges(edges, n)


def spatial_pairs(xyz, edges):
    """Deterministic maximal matching, followed by length-three augmentations.

    Unmatched elements remain singleton parents with a masked second slot.
    No long-range pairing is introduced to obtain a prescribed parent count.
    """
    n = len(xyz)
    edges = _canonical_edges(edges, n)
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(int(b))
        neighbours[b].append(int(a))
    for i in range(n):
        neighbours[i].sort(key=lambda j: (float(np.linalg.norm(xyz[j] - xyz[i])), j))
    mate = np.full(n, -1, dtype=np.int64)
    distances = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0], distances)) if len(edges) else []
    for idx in order:
        a, b = edges[idx]
        if mate[a] < 0 and mate[b] < 0:
            mate[a] = b
            mate[b] = a
    for a in range(n):
        if mate[a] >= 0:
            continue
        repaired = False
        for b in neighbours[a]:
            c = mate[b]
            if c < 0:  # Also handles any newly available direct neighbour.
                mate[a] = b
                mate[b] = a
                repaired = True
                break
            for d in neighbours[c]:
                if d != a and mate[d] < 0:
                    mate[a] = b
                    mate[b] = a
                    mate[c] = d
                    mate[d] = c
                    repaired = True
                    break
            if repaired:
                break
    pairs = []
    for i in range(n):
        j = int(mate[i])
        if j < 0:
            pairs.append((i, -1))
        elif i < j:
            # Stable spatial order for the two child-feature slots.
            ids = sorted((i, j), key=lambda t: tuple(xyz[t].tolist()) + (t,))
            pairs.append(tuple(ids))
    pairs = np.asarray(pairs, dtype=np.int64)
    parent = np.empty(n, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    for p, row in enumerate(pairs):
        for s, i in enumerate(row):
            if i >= 0:
                parent[i] = p
                slot[i] = s
    return pairs, parent, slot


def build_geometry(
    llh,
    area=None,
    *,
    mode="radius_knn",
    edges=None,
    triangles=None,
    k=6,
    max_distance_km=30.0,
    reference_lonlat=None,
):
    xyz = project_llh(llh, reference_lonlat)
    n = len(xyz)
    area = (
        np.ones(n) if area is None else np.asarray(area, dtype=np.float64).reshape(-1)
    )
    if area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Area must contain one finite positive value per element")
    if mode == "triangles":
        if triangles is None or len(triangles) != n:
            raise ValueError(
                "Provide ordered triangle connectivity for every input element"
            )
        edges = triangle_edges(triangles)
    elif mode == "edges":
        if edges is None:
            raise ValueError("mode=edges requires an adjacency table")
        edges = _canonical_edges(edges, n)
    elif mode == "radius_knn":
        edges = radius_knn_edges(xyz, k, max_distance_km)
    else:
        raise ValueError("geometry_mode must be triangles, edges, or radius_knn")
    pairs, parent, slot = spatial_pairs(xyz, edges)
    safe = pairs.clip(min=0)
    weights = area[safe] * (pairs >= 0)
    coarse_area = weights.sum(1)
    coarse_xyz = (xyz[safe] * weights[:, :, None]).sum(1) / coarse_area[:, None]
    pair_distance = np.linalg.norm(xyz[safe[:, 0]] - xyz[safe[:, 1]], axis=1)[
        pairs[:, 1] >= 0
    ]
    digest = hashlib.sha256()
    for arr in (
        np.asarray(llh, dtype="<f8"),
        area.astype("<f8"),
        edges.astype("<i8"),
        pairs.astype("<i8"),
    ):
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    if reference_lonlat is not None:
        digest.update(np.asarray(reference_lonlat, dtype="<f8").tobytes())
    digest.update(GEOMETRY_VERSION.encode())
    digest.update(mode.encode())
    metadata = dict(
        version=GEOMETRY_VERSION,
        geometry_hash=digest.hexdigest(),
        adjacency_mode=mode,
        approximate_adjacency=(mode == "radius_knn"),
        fine_tokens=n,
        coarse_tokens=len(pairs),
        singleton_parents=int(np.sum(pairs[:, 1] < 0)),
        edges=len(edges),
        pair_distance_km_max=float(pair_distance.max()) if len(pair_distance) else 0.0,
        pair_distance_km_median=(
            float(np.median(pair_distance)) if len(pair_distance) else 0.0
        ),
        projection_reference_lonlat=None if reference_lonlat is None else list(reference_lonlat),
        coordinate_system="regional azimuthal equidistant east/north; positive depth km",
        knn_k=int(k),
        max_neighbor_distance_km=float(max_distance_km),
    )
    return FaultGeometry(
        xyz, coarse_xyz, area, coarse_area, edges, pairs, parent, slot, metadata
    ).validate()


def _read_array(path, key=None):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            if key is None:
                if len(f.files) != 1:
                    raise ValueError("Specify an array key for a multi-array npz")
                key = f.files[0]
            return f[key].copy()
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path)


def load_geometry(params):
    """Build from configured sources and persist a content-addressed geometry cache."""
    llh_path = Path(params.llh_path)
    llh = _read_array(llh_path)
    expected = int(params.fea_len)
    if llh.shape != (expected, 3):
        raise ValueError(
            f"Fault geometry {llh.shape} disagrees with fea_len={expected}"
        )
    area_path = getattr(params, "area_path", str(llh_path.parent / "area"))
    area = _read_array(area_path)
    mode = getattr(params, "geometry_mode", "radius_knn")
    edges = triangles = None
    if mode == "triangles":
        triangles = _read_array(
            params.triangles_path, getattr(params, "triangles_key", None)
        )
    elif mode == "edges":
        edges = _read_array(
            params.adjacency_path, getattr(params, "adjacency_key", None)
        )
    geometry = build_geometry(
        llh,
        area,
        mode=mode,
        edges=edges,
        triangles=triangles,
        k=int(getattr(params, "geometry_knn_k", 6)),
        max_distance_km=float(getattr(params, "geometry_max_distance_km", 30.0)),
        reference_lonlat=getattr(params, "projection_reference_lonlat", None),
    )
    geometry.metadata["llh_path"] = str(llh_path.resolve())
    geometry.metadata["area_path"] = str(Path(area_path).resolve())
    cache_dir = getattr(params, "geometry_cache_dir", None)
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / (geometry.metadata["geometry_hash"] + ".npz")
        arrays = {
            name: getattr(geometry, name)
            for name in (
                "fine_xyz_km",
                "coarse_xyz_km",
                "fine_area",
                "coarse_area",
                "edge_index",
                "pairs",
                "parent",
                "slot",
            )
        }
        if not path.exists():
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".npz", delete=False
            ) as f:
                temp = Path(f.name)
                np.savez_compressed(
                    f,
                    **arrays,
                    metadata=np.array(json.dumps(geometry.metadata, sort_keys=True)),
                )
            os.replace(temp, path)
        else:
            with np.load(path, allow_pickle=False) as f:
                stored = json.loads(str(f["metadata"]))
                if stored["geometry_hash"] != geometry.metadata["geometry_hash"]:
                    raise ValueError("Geometry cache hash mismatch")
                for key, value in arrays.items():
                    if not np.array_equal(f[key], value):
                        raise ValueError(f"Geometry cache content mismatch: {key}")
        geometry.metadata["cache_path"] = str(path.resolve())
    return geometry


def regional_depth_weights(depth, region):
    depth = np.asarray(depth, dtype=np.float64)
    if region in ("cascadia", "alaska"):
        w = np.exp(-((depth - 35.0) ** 2) / (2 * 20.0**2))
    elif region == "nankai":
        w = np.maximum(
            np.exp(-((depth - 15.0) ** 2) / (2 * 20.0**2)),
            np.exp(-((depth - 45.0) ** 2) / (2 * 20.0**2)),
        )
    elif region == "newzealand":
        w = np.maximum(
            np.maximum(
                np.exp(-((depth - 7.5) ** 6) / (2 * 7.5**6)),
                np.exp(-((depth - 40.0) ** 6) / (2 * 11.0**6)),
            ),
            np.where(depth < 45.0, 0.8, 0.3),
        )
    else:
        raise ValueError(f"No depth-weight profile is defined for region={region!r}")
    return w.astype(np.float32)


@dataclass
class TensorFaultGeometry:
    """External, non-trainable geometry; no region-sized model parameters."""
    fine_xyz: object
    coarse_xyz: object
    pairs: object
    parent: object
    slot: object
    depth_weights: object
    metadata: dict
    fine_area: object = None
    coarse_area: object = None
    edge_index: object = None

    @classmethod
    def from_geometry(cls, geometry, region, device="cpu", scales=(100., 100., 20.)):
        import torch
        geometry.validate()
        scale = torch.as_tensor(scales, dtype=torch.float32, device=device)
        if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid coordinate scales")
        tensor = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
        return cls(
            tensor(geometry.fine_xyz_km, torch.float32) / scale,
            tensor(geometry.coarse_xyz_km, torch.float32) / scale,
            tensor(geometry.pairs, torch.long), tensor(geometry.parent, torch.long),
            tensor(geometry.slot, torch.long),
            tensor(joint_depth_weights(geometry.fine_xyz_km[:, 2], region), torch.float32),
            dict(geometry.metadata, region=region, coordinate_scales=list(scales)),
            tensor(geometry.fine_area, torch.float32),
            tensor(geometry.coarse_area, torch.float32),
            tensor(geometry.edge_index, torch.long),
        )


def joint_depth_weights(depth, region):
    """Joint-v1 profiles. Legacy regional_depth_weights stays unchanged."""
    z = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("Expected finite positive-down depths in km")
    centers = {"cascadia": 35., "alaska": 45., "nankai": 45.}
    if region in centers:
        w = np.exp(-0.5 * ((z - centers[region]) / 20.) ** 2)
    elif region == "newzealand":
        w = np.maximum(np.exp(-0.5 * ((z - 7.5) / 7.5) ** 6),
                       np.where(z < 45., .8, .3))
    else:
        raise ValueError(f"Unknown joint region: {region}")
    return w.astype(np.float32)
