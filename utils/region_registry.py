"""多区域 token 布局注册表（多区域训练用）。

把每个区域的 ``build_layout`` 结果（槽位映射 / unpad 索引 / 平均经纬度）连同网格采样坐标
缓存成 npz，供多区域训练时按 ``region`` 取用。

网格坐标采用**保长宽比的 letterbox 归一化**：以区域内最长方向的物理长度为 1，居中放进
[-1, 1] 的网格坐标（square cells）。若像单区域那样把每个区域的 bbox 各自拉伸填满网格，
514×1242 km 的 cascadia 和 1123×1107 km 的 alaska 会被归一化到同一个方形，几何比例被破坏，
共享同一张 ``pos_embed_grid`` 时语义就不一致了。
"""

import os
import numpy as np

from utils.fault_tokens import build_layout

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32


DEFAULT_KEYS = ("data_path", "fault_path", "llh_path", "mean_path", "std_path",
                "tremor_path", "mask_key")


def region_config(params, region):
    """yaml 里 ``region_configs.<region>`` 那一节（没有就返回空 dict）。"""
    cfgs = getattr(params, "region_configs", None) or {}
    cfg = cfgs.get(region) if hasattr(cfgs, "get") else None
    return dict(cfg) if cfg else {}


def region_data_paths(params, region):
    """区域的数据 / 断层 / 归一化 / tremor / mask_key。

    全部来自 yaml 的 ``region_configs.<region>``（一个区域一节，新增区域不用改代码）；
    只有在缺这一节时才退回 ``<data_root>/<region>/<data_suffix>`` 的老约定。
    """
    cfg = region_config(params, region)
    root = str(getattr(params, "data_root", "../Data"))
    suffix = str(getattr(params, "data_suffix", "slip_potency_smooth"))
    base = os.path.join(root, region)
    default = {
        "data_path": os.path.join(base, suffix),
        "fault_path": os.path.join(base, "fault"),
        "llh_path": os.path.join(base, "fault", "llh"),
        "mean_path": os.path.join(base, "norm_value", "mean.npy"),
        "std_path": os.path.join(base, "norm_value", "std.npy"),
        "tremor_path": os.path.join(base, "tremor"),
        "mask_key": region,
    }
    out = {k: str(cfg.get(k) or default[k]) for k in DEFAULT_KEYS}
    missing = [k for k in ("data_path", "fault_path", "llh_path") if not os.path.exists(out[k])]
    if missing:
        raise FileNotFoundError(
            "区域 %r 的路径不存在：%s（检查 config 里的 region_configs.%s）"
            % (region, {k: out[k] for k in missing}, region))
    return out


def letterbox_coords(lon, lat, grid_h, grid_w):
    """经纬度 → 网格采样坐标 (1,1,S,2)，[-1,1]，保长宽比（单位 = 该区域最长方向）。"""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    k = float(np.cos(np.deg2rad(lat.mean())))
    x_km = lon * KM_PER_DEG_LON * k
    y_km = lat * KM_PER_DEG_LAT
    span = max(float(x_km.max() - x_km.min()), float(y_km.max() - y_km.min()), 1e-6)
    u = (x_km - 0.5 * (x_km.max() + x_km.min())) / span          # ∈ [-0.5, 0.5]
    v = (y_km - 0.5 * (y_km.max() + y_km.min())) / span
    return np.stack([u * 2.0, v * 2.0], axis=-1).astype(np.float32)


def build_region(params, region, patch_size=2, jump_km=30.0, order="strip",
                 pairing="diagonal", grid=(48, 48), verbose=True):
    """构造单个区域的布局与网格坐标（不做缓存）。"""
    paths = region_data_paths(params, region)
    fault_path = paths["fault_path"]
    fea_len = len(np.loadtxt(os.path.join(fault_path, "lat"), ndmin=2))
    lay = build_layout(fault_path, fea_len, patch_size=int(patch_size),
                       jump_km=float(jump_km), order=str(order), pairing=str(pairing),
                       verbose_order=True)   # 打印每个区域最终采用的编号顺序（native / banded）
    lon = lay["token_lonlat"][:, 0]
    lat = lay["token_lonlat"][:, 1]
    grid_h, grid_w = int(grid[0]), int(grid[1])
    coords = letterbox_coords(lon, lat, grid_h, grid_w)
    out = {
        "region": region,
        "fea_len": int(fea_len),
        "n_tokens": int(lay["n_tokens"]),
        "patch_size": int(patch_size),
        "slot_index": np.asarray(lay["slot_to_element"], dtype=np.int64),   # (n_tokens, patch_size)
        "unpad_index": np.asarray(lay["unpad_index"], dtype=np.int64),      # (fea_len,)
        "grid_coords": coords.reshape(1, 1, -1, 2),                         # (1,1,S,2)
        "token_lonlat": np.asarray(lay["token_lonlat"], dtype=np.float64),
        "n_strips": int(lay["n_strips"]),
        "grid_h": grid_h,
        "grid_w": grid_w,
        **{k: v for k, v in paths.items()},
    }
    if verbose:
        ext_km = (
            float(lon.max() - lon.min()) * KM_PER_DEG_LON * np.cos(np.deg2rad(lat.mean())),
            float(lat.max() - lat.min()) * KM_PER_DEG_LAT,
        )
        print("[region] %-11s 单元 %4d → token %4d | bbox %.0f×%.0f km | 网格 %d×%d (单元 %.0f km)"
              % (region, fea_len, out["n_tokens"], ext_km[0], ext_km[1], grid_h, grid_w,
                 max(ext_km) / max(grid_h, grid_w) * 1.0))
    return out


def build_regions(params, regions, cache_dir=None, **kwargs):
    """批量构造（带 npz 缓存，命中即秒读）。返回 {region: dict}。"""
    regions = list(regions)
    grid = tuple(kwargs.get("grid", (48, 48)))
    patch_size = int(kwargs.get("patch_size", 2))
    out = {}
    for r in regions:
        if cache_dir is None:
            out[r] = build_region(params, r, **kwargs)
            continue
        os.makedirs(cache_dir, exist_ok=True)
        # 缓存键必须包含"会改变布局"的所有参数，否则改了 pairing/order 会静默复用旧布局
        f = os.path.join(cache_dir, "region_%s_p%d_%s_%s_j%g_g%dx%d.npz" % (
            r, patch_size, str(kwargs.get("pairing", "diagonal")),
            str(kwargs.get("order", "strip")), float(kwargs.get("jump_km", 30.0)),
            grid[0], grid[1]))
        if os.path.isfile(f):
            z = np.load(f, allow_pickle=False)
            d = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
            d["region"] = r
            for k, v in region_data_paths(params, r).items():
                d.setdefault(k, v)
            out[r] = d
            continue
        d = build_region(params, r, **kwargs)
        np.savez(f, **{k: v for k, v in d.items() if isinstance(v, (int, float, str, np.ndarray))})
        out[r] = d
    n = [d["n_tokens"] for d in out.values()]
    print("[region] %d 个区域: tokens %s (max %d)" % (len(out), n, max(n)))
    return out
