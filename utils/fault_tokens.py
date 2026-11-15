"""按断层网格的"条带(strip)"结构对齐 patch/token 边界（方案 A）。

背景
----
数据表里的 3339 个断层三角形是按**沿倾角的条带**排列的：条带内纬度基本不变、
深度单调增加（浅 → 深）、经度同步向东；条带之间向南平移约 0.2°，并且每个条带
在末尾会从"最深"一下跳回"最浅"（跳变中位 ~276 km，最大 ~417 km）。

原来的 token 化是 `Conv1d(kernel=patch_size, stride=patch_size)` 直接作用在文件行序上，
于是 `fea_len // patch_size` 个 token 里约 6% 会横跨条带接缝，把相距几百公里的两个区域
混进同一个 token（实测 49/1113 个 token 的 3 个单元两两距离 >30 km）。

本模块把每个条带补齐到 patch_size 的整数倍（末尾复制该条带最后一个单元），
再按 patch_size 切 token —— 于是**每个 token 都完整落在同一个条带内**，代价是
token 数从 1113 增加到 1137（+2.2%）。
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def fault_centers_km(fault_path, n_elements=None):
    """读取断层三角形顶点，返回质心的局部平面坐标 (n, 2)，单位 km。"""
    lat = np.loadtxt(str(fault_path) + "/lat", ndmin=2)
    lon = np.loadtxt(str(fault_path) + "/lon", ndmin=2)
    clat = lat.mean(axis=1) if lat.ndim == 2 else lat
    clon = lon.mean(axis=1) if lon.ndim == 2 else lon
    if n_elements is not None and clat.size != n_elements:
        raise ValueError(f"断层单元数 {clat.size} 与 fea_len {n_elements} 不一致")
    x = np.deg2rad(clon) * EARTH_RADIUS_KM * np.cos(np.deg2rad(clat.mean()))
    y = np.deg2rad(clat) * EARTH_RADIUS_KM
    return np.stack([x, y], axis=1)


def median_nn_km(centers):
    """每个点的最近邻距离的中位数（km）—— 用来自动定"带厚"和"连贯性阈值"。"""
    c = np.asarray(centers, dtype=np.float64)
    if len(c) < 2:
        return 0.0
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def order_coherence(centers, order_idx=None, thr_factor=3.0):
    """1-D 编号顺序的"空间连贯性"：相邻编号距离 < thr_factor×最近邻中位距 的比例。

    1.0 = 完美（相邻编号总是空间邻居）；接近 0 = 编号与空间无关（RoPE/DownSample 会看到错误邻居）。
    """
    c = np.asarray(centers, dtype=np.float64)
    if order_idx is not None:
        c = c[np.asarray(order_idx, dtype=np.int64)]
    if len(c) < 3:
        return 1.0
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    thr = max(thr_factor * median_nn_km(np.asarray(centers, dtype=np.float64)), 1e-6)
    return float(np.mean(step < thr))


def banded_order(centers, band_km=None, thr_factor=3.0):
    """按几何重新编号：PCA 主轴分带（跨走向），带内沿走向排序，带间蛇形折返。

    对"文件行序与空间无关"的区域（alaska / nankai / newzealand），
    能把相邻编号距离从 260~420 km 压到 ~15 km，同时让 DownSample 的配对变成真邻居。
    ``band_km`` 默认 = 3 × 最近邻中位距（约 3 个 token 厚的一条带）。
    """
    c = np.asarray(centers, dtype=np.float64)
    if len(c) < 4:
        return np.arange(len(c), dtype=np.int64)
    nn = median_nn_km(c)
    band_km = float(band_km) if band_km else max(thr_factor * nn, 1e-6)
    d0 = c - c.mean(axis=0)
    _, vec = np.linalg.eigh(d0.T @ d0)
    e_along, e_across = vec[:, -1], vec[:, -2]        # 最长方向 / 次长方向
    s = d0 @ e_along
    a = d0 @ e_across
    nb = int(np.floor((a.max() - a.min()) / band_km)) + 1
    band = np.clip(((a - a.min()) // band_km).astype(np.int64), 0, nb - 1)
    seq = []
    for b in range(nb):
        m = np.where(band == b)[0]
        if m.size == 0:
            continue
        m = m[np.argsort(s[m])]
        seq.extend((m[::-1] if b % 2 else m).tolist())
    return np.asarray(seq, dtype=np.int64)


def strip_ids(centers, jump_km=30.0):
    """按相邻质心距离 > jump_km 的位置把单元序列切成连续条带，返回每个单元的条带号。"""
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cuts = np.where(step > float(jump_km))[0] + 1
    return np.repeat(np.arange(cuts.size + 1), np.diff(np.concatenate([[0], cuts, [centers.shape[0]]])))


def _mean_intra_km(centers, groups):
    """一组 token（每个 token 是若干单元下标）内部的平均两两距离。"""
    vals = []
    for g in groups:
        if len(g) < 2:
            continue
        d = [np.linalg.norm(centers[g[i]] - centers[g[j]])
             for i in range(len(g)) for j in range(i + 1, len(g))]
        vals.append(float(np.mean(d)))
    return float(np.mean(vals)) if vals else np.inf


def triangle_geometry(fault_path):
    """重建三角网格的拓扑：顶点索引、每个三角形的 3 个顶点、边长（km）。"""
    lat = np.loadtxt(str(fault_path) + "/lat", ndmin=2)
    lon = np.loadtxt(str(fault_path) + "/lon", ndmin=2)
    n = lat.shape[0]
    key_to_id, tris = {}, []
    for i in range(n):
        ids = []
        for k in range(3):
            key = (round(float(lat[i, k]), 6), round(float(lon[i, k]), 6))
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
            ids.append(key_to_id[key])
        tris.append(tuple(ids))
    coord = np.zeros((len(key_to_id), 2))
    for key, idx in key_to_id.items():
        coord[idx] = key                                  # (lat, lon)
    lat0 = np.deg2rad(coord[:, 0].mean())
    xy = np.stack([np.deg2rad(coord[:, 1]) * EARTH_RADIUS_KM * np.cos(lat0),
                   np.deg2rad(coord[:, 0]) * EARTH_RADIUS_KM], axis=1)
    return {"tris": tris, "xy": xy, "lat": lat, "lon": lon}


def diagonal_pairs(fault_path, n_elements, verbose=False):
    """把「共享同一条斜边的两个三角形」配对成一个四边形。

    判定：每个三角形取**最长边**作为候选斜边；若它对面的那个三角形也以这条边为最长边，
    则两者互为配对（= 同一个四边形的两半）。没有这样的伙伴时返回 -1（该三角形单独成 token，补 padding）。

    返回 (pairs: list[(i, j)], stats: dict)，j == -1 表示配不上。
    """
    geo = triangle_geometry(fault_path)
    tris = geo["tris"]
    if len(tris) != int(n_elements):
        raise ValueError(f"三角形数 {len(tris)} 与 fea_len {n_elements} 不一致")
    xy = geo["xy"]

    def elen(e):
        return float(np.linalg.norm(xy[e[0]] - xy[e[1]]))

    edges = [[(a, b), (b, c), (c, a)] for (a, b, c) in tris]
    L = np.array([[elen(e) for e in es] for es in edges])
    diag_idx = np.argmax(L, axis=1)                        # 每个三角形的最长边

    edge_to_tris = {}
    for t, es in enumerate(edges):
        for e in es:
            edge_to_tris.setdefault(tuple(sorted(e)), []).append(t)

    partner = -np.ones(len(tris), dtype=np.int64)
    for t in range(len(tris)):
        e = tuple(sorted(edges[t][diag_idx[t]]))
        others = [u for u in edge_to_tris[e] if u != t]
        if len(others) == 1:
            u = others[0]
            u_diag = tuple(sorted(edges[u][diag_idx[u]]))
            if u_diag == e:                                # 双方都以它为最长边 -> 同一四边形
                partner[t] = u

    pairs, seen = [], set()
    for t in range(len(tris)):
        if t in seen:
            continue
        u = int(partner[t])
        if u >= 0 and u not in seen:
            pairs.append((t, u)); seen.update((t, u))
        else:
            pairs.append((t, -1)); seen.add(t)

    stats = {
        "n_pairs": int(sum(1 for _, j in pairs if j >= 0)),
        "n_unmatched": int(sum(1 for _, j in pairs if j < 0)),
        "diag_len_median_km": float(np.median([L[t][diag_idx[t]] for t in range(len(tris))])),
        "side_len_median_km": float(np.median(np.sort(L, axis=1)[:, :2])),
    }
    if verbose:
        print(f"[diagonal_pairs] {stats['n_pairs']} 对 + {stats['n_unmatched']} 个单独 "
              f"(斜边中位 {stats['diag_len_median_km']:.1f} km, 网格边 {stats['side_len_median_km']:.1f} km)")
    return pairs, stats


def paired_groups(fault_path, n_elements, patch_size=2, verbose=False, strict=True, mode=None):
    """按网格拓扑把三角形分组：先"共斜边配对成四边形"，再把相邻四边形合并到一个 token。

    * ``patch_size=2`` → 一个 token = 一个四边形（2 个三角形，共享斜边）；
    * ``patch_size=4`` → 一个 token = **两个共享网格边的相邻四边形**（4 个三角形）；
    * 其它偶数同理（每 2 个三角形补一个四边形，再按网格邻接贪心合并）。

    ``strict`` 决定"怎样的两片算一个四边形"：
      * ``True``（``pairing="diagonal"``）：必须**双方都以共享边为最长边**（严格四边形二分网格）；
      * ``False``（``pairing="diagonal_relaxed"``）：只要**共享任意一条边**就配，
        贪心按"共享边长 → 短"优先。不规则/三角网格里能把"配不上"的比例从 ~27% 降到 ~10%，
        代价是个别 token 形状不那么像规整四边形（仍是两片相邻三角形）。

    返回 (groups, stats)：
      groups : list[tuple[int, ...]]，每个是 patch_size 个三角形编号（配不齐的更短，由调用方补 padding）
      stats  : 统计（邻接比例、组内最大间距等）
    """
    mode = ("strict" if strict else "greedy") if mode is None else str(mode)
    if mode not in ("strict", "fill", "greedy"):
        raise ValueError(f"mode 只能是 strict/fill/greedy，收到 {mode!r}")
    if int(patch_size) % 2 or int(patch_size) < 2:
        raise ValueError(f"pairing='diagonal' 需要偶数 patch_size（2/4/6…），收到 {patch_size}")
    patch_size = int(patch_size)
    geo = triangle_geometry(fault_path)
    tris, xy = geo["tris"], geo["xy"]
    if len(tris) != int(n_elements):
        raise ValueError(f"三角形数 {len(tris)} 与 fea_len {n_elements} 不一致")

    def elen(e):
        return float(np.linalg.norm(xy[e[0]] - xy[e[1]]))

    edges = [[(a, b), (b, c), (c, a)] for (a, b, c) in tris]
    L = np.array([[elen(e) for e in es] for es in edges])
    diag_idx = np.argmax(L, axis=1)                     # 最长边 = 斜边

    edge_to_tris = {}
    for t, es in enumerate(edges):
        for i, e in enumerate(es):
            edge_to_tris.setdefault(tuple(sorted(e)), []).append((t, i))

    # ---- 1) 配对成四边形 ----
    partner = -np.ones(len(tris), dtype=np.int64)
    if mode == "strict":
        # 严格规则：共享边必须是**双方的最长边**（斜边）
        for t in range(len(tris)):
            e = tuple(sorted(edges[t][diag_idx[t]]))
            others = [u for (u, _) in edge_to_tris[e] if u != t]
            if len(others) == 1:
                u = others[0]
                if tuple(sorted(edges[u][diag_idx[u]])) == e:
                    partner[t] = u
    else:
        # 放宽规则：共享任意一条边就能配；按边长从短到长贪心（短边 = 更贴合的四边形）
        def _greedy(cand, used_pair):
            for _, a, b in sorted(cand):
                if used_pair[a] or used_pair[b]:
                    continue
                partner[a], partner[b] = b, a
                used_pair[a] = used_pair[b] = True

        cand = []
        for e, lst in edge_to_tris.items():
            if len(lst) == 2:
                cand.append((float(np.linalg.norm(xy[e[0]] - xy[e[1]])), lst[0][0], lst[1][0]))
        if mode == "greedy":
            _greedy(cand, np.zeros(len(tris), dtype=bool))
        else:                                   # mode == "fill"
            # 先保留严格配对的"规整四边形"，只对剩下配对失败的三角做贪心补配
            for t in range(len(tris)):
                e = tuple(sorted(edges[t][diag_idx[t]]))
                others = [u for (u, _) in edge_to_tris[e] if u != t]
                if len(others) == 1:
                    u = others[0]
                    if tuple(sorted(edges[u][diag_idx[u]])) == e:
                        partner[t] = u
            done = partner >= 0
            rest = [c for c in cand if not done[c[1]] and not done[c[2]]]
            _greedy(rest, done.copy())

    quads, seen = [], set()
    for t in range(len(tris)):
        if t in seen:
            continue
        u = int(partner[t])
        if u >= 0 and u not in seen:
            quads.append([t, u]); seen.update((t, u))
        else:
            quads.append([t]); seen.add(t)

    tri_cent = np.array([xy[list(tr)].mean(axis=0) for tr in tris])   # 每个三角形的质心
    cent = np.array([tri_cent[q].mean(axis=0) for q in quads])       # 每个四边形的质心
    n_quad = len(quads)
    k = patch_size // 2                                  # 一个 token 里放几个四边形

    if k == 1:
        groups = [tuple(q) for q in quads]
        padded_groups = 0
        adj_ok = 1.0
    else:
        # ---- 2) 四边形之间的网格邻接（共享**非斜边**的边）----
        def grid_edges(q):
            out = set()
            for t in q:
                for i, e in enumerate(edges[t]):
                    if i != diag_idx[t]:
                        out.add(tuple(sorted(e)))
            return out

        qedges = [grid_edges(q) for q in quads]
        edge_to_quads = {}
        for qi, qe in enumerate(qedges):
            for e in qe:
                edge_to_quads.setdefault(e, []).append(qi)
        adj = [set() for _ in range(n_quad)]
        for e, qs in edge_to_quads.items():
            if len(qs) == 2:
                adj[qs[0]].add(qs[1]); adj[qs[1]].add(qs[0])

        used = np.zeros(n_quad, bool)
        groups, adj_ok_count, padded_groups = [], 0, 0
        for qi in range(n_quad):
            if used[qi]:
                continue
            grp = [qi]; used[qi] = True
            while len(grp) < k:
                cands = [j for gi in grp for j in adj[gi] if not used[j]]
                adjacent = True
                if not cands:                                  # 没有相邻的，退化为最近的
                    rest = np.where(~used)[0]
                    if rest.size == 0:
                        break
                    cands, adjacent = list(rest), False
                j = min(set(cands), key=lambda x: float(np.linalg.norm(cent[grp[-1]] - cent[x])))
                grp.append(j); used[j] = True
                if not adjacent:
                    padded_groups += 0                     # 只统计邻接比例
            if len(grp) == k and all(j in adj[grp[0]] or j == grp[0] for j in grp):
                adj_ok_count += 1
            groups.append(tuple(t for gi in grp for t in quads[gi]))
        adj_ok = adj_ok_count / max(1, len(groups))
        padded_groups = 0

    # 组内空间间距统计
    intra = []
    for g in groups:
        if len(g) > 1:
            p_ = tri_cent[list(g)]
            d = np.linalg.norm(p_[:, None] - p_[None], axis=-1)
            intra.append(float(d.max()))
    stats = {
        "patch_size": patch_size,
        "pairing_mode": str(mode),
        "n_quads": int(n_quad),
        "n_groups": len(groups),
        "n_full": int(sum(1 for g in groups if len(g) == patch_size)),
        "n_short": int(sum(1 for g in groups if len(g) < patch_size)),
        "adjacent_ratio": float(adj_ok),
        "intra_km_median": float(np.median(intra)) if intra else 0.0,
        "intra_km_p99": float(np.percentile(intra, 99)) if intra else 0.0,
        "intra_km_max": float(np.max(intra)) if intra else 0.0,
        "diag_len_median_km": float(np.median([L[t][diag_idx[t]] for t in range(len(tris))])),
        "side_len_median_km": float(np.median(np.sort(L, axis=1)[:, :2])),
    }
    if verbose:
        print(f"[paired_groups] patch={patch_size}: {stats['n_groups']} 组 "
              f"({stats['n_full']} 组满员 / {stats['n_short']} 组缺员) | 相邻比例 {stats['adjacent_ratio']:.2f} | "
              f"组内最大间距 中位 {stats['intra_km_median']:.1f} / max {stats['intra_km_max']:.0f} km")
    return groups, stats


def build_layout(fault_path, n_elements, patch_size=3, jump_km=30.0,
                 optimize_phase=True, order="serpentine", pairing="phase", verbose_order=False):
    """构造"对齐 token 边界"的槽位映射。

    ``pairing`` 决定"哪些 patch 组成一个 token"：
      * ``"phase"``（默认）：条带内按 patch_size 连续切块，并用相位优化对齐四边形；
      * ``"diagonal"``：**共享同一条斜边的两个三角形**配成一个 token（= 一个四边形）。
        判定用最长边：每个三角形的最长边即斜边，若对面三角形也以同一条边为最长边则配对成功；
        配不上的三角形单独成 token，缺的位置补 padding（``pad_mask=True``，模型输出时 unpad 丢弃）。
        只支持 ``patch_size=2``。

    两个对齐：
    * ``optimize_phase``：每条条带尝试 patch_size 种起始相位（在条带内平移 token 边界），
      取组内平均间距最小的那个。``patch_size=2`` 时能把每个 token 对齐到"一个四边形
      = 2 个三角形"（组内平均间距 15.8 km -> 7.1 km）。
    * ``order``：token **编号顺序**。
        - ``"strip"``      ：按条带依次排（原始顺序）。每条条带内部平滑，但从"最深端"跳回
                             下一条条带的"最浅端"——相邻 token 编号可能相距几百公里。
        - ``"serpentine"`` ：蛇形走位。每条条带从"离上一条末尾最近的一端"开始，
                             于是条带之间的切换也是空间相邻的（>50 km 的跳变从 69 处降到 1 处，
                             中位/最大 = 12.0 / 244 km）。
      编号平滑对 RoPE（只编码 token 序号距离）很重要：相邻序号≈相邻位置。

    返回 dict（numpy 数组 / 标量）：
      slot_to_element : (n_slots,) 槽位 → 断层单元号（补齐位 = 该条带端点的复制）
      pad_mask        : (n_slots,) bool，True = 补齐位（loss/可视化应忽略）
      strip_of_slot   : (n_slots,) 槽位所属条带
      slot_of_element : (n_elements,) 单元 → 槽位
      unpad_index     : (n_elements,) 从槽位空间取回单元顺序（长度 = fea_len）
      token_centroid  : (n_tokens, 2) 每个 token 的质心（km），用于画图/平滑度诊断
      token_lonlat    : (n_tokens, 2) 每个 token 的平均经纬度 (lon, lat)
      token_strip     : (n_tokens,) 每个 token 的条带号
      token_order     : (n_tokens,) 蛇形重排用的排列（order="strip" 时为单位排列）
      n_tokens / n_slots / n_strips / strip_lengths / n_pad / phases
    """
    patch_size = int(patch_size)
    if order not in ("strip", "serpentine", "raster", "banded", "auto"):
        raise ValueError(f"order 只能是 strip/serpentine/raster/banded/auto，收到 {order!r}")
    order_final = order
    centers = fault_centers_km(fault_path, n_elements)
    strips = strip_ids(centers, jump_km)

    token_slots, token_pads, token_strips, phases = [], [], [], []
    diag_stats = None
    if pairing in ("diagonal", "diagonal_fill", "diagonal_relaxed"):
        # 先共边配对成四边形，再把相邻四边形合并成一个 token（支持 2/4/6…）
        _mode = {"diagonal": "strict", "diagonal_fill": "fill",
                 "diagonal_relaxed": "greedy"}[pairing]
        groups, diag_stats = paired_groups(fault_path, n_elements, patch_size, verbose=True, mode=_mode)
        rows, pads = [], []
        for g in groups:
            g = list(g)
            n_pad = patch_size - len(g)
            rows.append(g + [g[-1]] * n_pad)                        # 缺的位用组内最后一个单元补
            pads.append([False] * len(g) + [True] * n_pad)
        rows = np.asarray(rows, dtype=np.int64); pads = np.asarray(pads, dtype=bool)
        token_slots.append(rows)
        token_pads.append(pads)
        token_strips.append(strips[rows[:, 0]].astype(np.int64))
        phases.append(0)
    else:
        for sid in range(int(strips.max()) + 1):
            idx = np.where(strips == sid)[0]
            # ---- 选相位：p = "token 从条带内第 p 个单元开始配对" ----
            best_phase, best_key = 0, None
            if optimize_phase and patch_size > 1:
                for ph in range(patch_size):
                    m = idx.size - ph - (idx.size - ph) % patch_size
                    groups = idx[ph:ph + m].reshape(-1, patch_size) if m > 0 else np.empty((0, patch_size), dtype=idx.dtype)
                    key = (_mean_intra_km(centers, groups), ph)   # 距离优先，其次 padding 少
                    if best_key is None or key < best_key:
                        best_phase, best_key = ph, key
            # ---- 槽位 = [前置补齐 p] + [全部单元] + [末尾补齐] ----
            head = np.repeat(idx[0], best_phase)
            n_tail = (-(best_phase + idx.size)) % patch_size
            tail = np.repeat(idx[-1], n_tail)
            slots = np.concatenate([head, idx, tail])
            pads = np.concatenate([np.ones(head.size, bool), np.zeros(idx.size, bool), np.ones(tail.size, bool)])
            token_slots.append(slots.reshape(-1, patch_size))
            token_pads.append(pads.reshape(-1, patch_size))
            token_strips.append(np.full(slots.size // patch_size, sid, dtype=np.int64))
            phases.append(best_phase)

    token_slots = np.concatenate(token_slots)
    token_pads = np.concatenate(token_pads)
    token_strips = np.concatenate(token_strips)
    token_centroid = centers[token_slots].mean(axis=1)          # (n_tokens, 2)
    # 每个 token 的**平均经纬度**（其 patch_size 个单元质心的均值），供经纬度网格位置编码使用
    _lat = np.loadtxt(str(fault_path) + "/lat", ndmin=2).mean(axis=1)
    _lon = np.loadtxt(str(fault_path) + "/lon", ndmin=2).mean(axis=1)
    token_lonlat = np.stack([_lon[token_slots].mean(axis=1),
                             _lat[token_slots].mean(axis=1)], axis=1)   # (n_tokens, 2) = (lon, lat)

    # ---- token 编号顺序：蛇形（每条条带从离上一条末尾最近的一端开始）----
    order_idx = np.arange(len(token_slots))
    if order == "raster":
        # 二维"栅格"顺序：沿走向（纬度）分箱做主序，箱内按沿倾角坐标排序 ——
        # 编号在纬度方向单调、在倾角方向也单调，像把 2D 网格按行读出来。
        row = np.round(token_centroid[:, 1] / 6.0).astype(np.int64)
        order_idx = np.lexsort((token_centroid[:, 0], -row))
    elif order in ("serpentine", "auto"):
        # auto 的"原生顺序"就用蛇形（对我们这批数据总是 ≥ strip）
        seq, prev = [], None
        for sid in np.unique(token_strips):
            tids = np.where(token_strips == sid)[0]
            first, last = token_centroid[tids[0]], token_centroid[tids[-1]]
            if prev is not None and np.linalg.norm(prev - last) < np.linalg.norm(prev - first):
                tids = tids[::-1]
            seq.extend(tids.tolist())
            prev = token_centroid[tids[-1]]
        order_idx = np.asarray(seq, dtype=np.int64)

    # ---- 按几何重新编号：banded（一律重排）/ auto（只在原生顺序不连贯时重排）----
    if order in ("banded", "auto"):
        geo_idx = banded_order(token_centroid)
        native_idx = order_idx.copy()
        native_score = order_coherence(token_centroid, native_idx)
        geo_score = order_coherence(token_centroid, geo_idx)
        if order == "banded":
            order_idx, order_final = geo_idx, "banded"
        else:                                       # auto：只有明显更好才换（避免无谓改布局）
            use_geo = geo_score > native_score + 0.02
            order_idx = geo_idx if use_geo else order_idx
            order_final = "banded" if use_geo else "native"
        if verbose_order:
            name = str(fault_path).rstrip("/").split("/")[-2]

            def _med_step(idx):
                c = token_centroid[idx] if idx is not None else token_centroid
                return float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)))

            print("[order] %-11s 原生 %.2f (相邻中位 %.0f km) | banded %.2f (相邻中位 %.0f km) → 采用 %s"
                  % (name, native_score, _med_step(native_idx), geo_score, _med_step(geo_idx), order_final))

    token_slots = token_slots[order_idx]
    token_pads = token_pads[order_idx]
    token_strips = token_strips[order_idx]
    token_centroid = token_centroid[order_idx]
    token_lonlat = token_lonlat[order_idx]

    # ---- 每个 token 的二维几何坐标 (u, v)，供 2-D 位置编码使用 ----
    # v = 条带序号（沿走向）；u = 沿倾角的弧长（以中位 token 间距为 1 个单位）
    spacing = float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    spacing = spacing if spacing > 1e-6 else 1.0
    token_uv = np.zeros((len(token_slots), 2), dtype=np.float64)
    token_uv[:, 1] = token_strips.astype(np.float64)          # 沿走向：条带号
    for sid in np.unique(token_strips):
        ids = np.where(token_strips == sid)[0]
        # 沿倾角：条带内按（沿走向的）位置排序后累加弧长；因为 snake/raster 可能反向，这里用 v 与 u 无关的重排
        c = token_centroid[ids]
        if len(ids) > 1:
            # 以质心在第一主轴上的投影排序，得到从浅到深的顺序，再累加相邻距离
            d = np.linalg.norm(np.diff(c, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(d)]) / spacing
        else:
            arc = np.zeros(1)
        token_uv[ids, 0] = arc

    slot_to_element = token_slots.ravel().astype(np.int64)
    pad_mask = token_pads.ravel()
    strip_of_slot = np.repeat(token_strips, patch_size)
    n_slots = slot_to_element.size
    if n_slots % patch_size:
        raise AssertionError("槽位数必须是 patch_size 的整数倍")

    slot_of_element = np.empty(int(n_elements), dtype=np.int64)
    slot_of_element[slot_to_element] = np.arange(n_slots)
    # 蛇形重排后，有效槽位在槽位空间里不再按单元号递增排列；这里把 unpad_index 排序成
    # "按单元号递增"，于是模型最后一步 index_select(-1, unpad_index) 仍然直接得到单元的原始顺序。
    valid = np.where(~pad_mask)[0].astype(np.int64)
    unpad_index = valid[np.argsort(slot_to_element[valid], kind="stable")]
    if unpad_index.size != int(n_elements):
        raise AssertionError("有效槽位数应与断层单元数一致")
    if not np.array_equal(slot_to_element[unpad_index], np.arange(int(n_elements))):
        raise AssertionError("unpad_index 排序后应恢复单元原始顺序")

    lengths = np.bincount(strips)
    return {
        "slot_to_element": slot_to_element,
        "slot_of_element": slot_of_element,
        "pad_mask": pad_mask,
        "strip_of_slot": strip_of_slot,
        "strip_of_element": strips.astype(np.int64),
        "unpad_index": unpad_index,
        "token_slot_index": token_slots,          # (n_tokens, patch_size)
        "token_pad_mask": token_pads,             # (n_tokens, patch_size)
        "token_centroid": token_centroid,         # (n_tokens, 2)
        "token_strip": token_strips,              # (n_tokens,)
        "token_order": order_idx,                 # (n_tokens,)
        "token_uv": token_uv,                     # (n_tokens, 2) 几何坐标 (沿倾角u, 沿走向v)
        "token_lonlat": token_lonlat,             # (n_tokens, 2) 平均经纬度 (lon, lat)
        "n_tokens": int(n_slots // patch_size),
        "n_slots": int(n_slots),
        "n_strips": int(strips.max() + 1),
        "strip_lengths": lengths,
        "n_pad": int(pad_mask.sum()),
        "phases": np.asarray(phases, dtype=np.int64),
        "pairing": pairing,
        "pairing_stats": diag_stats,
        "patch_size": patch_size,
        "jump_km": float(jump_km),
        "order": order,
        "order_final": order_final,
        "order_coherence": order_coherence(token_centroid),
    }



def element_token_id(layout):
    """每个断层单元所属 token 的编号（按 layout 的最终编号顺序），用于地图着色。"""
    ps = int(layout["patch_size"])
    tok_of_slot = np.repeat(np.arange(layout["n_tokens"], dtype=np.int64), ps)
    n_elem = int(layout["strip_of_element"].size)
    out = np.full(n_elem, -1, dtype=np.int64)
    out[layout["slot_to_element"]] = tok_of_slot
    return out


def smoothness_report(fault_path, n_elements, patch_size, jump_km=30.0,
                      orders=("strip", "raster", "serpentine"), pairing="phase"):
    """比较不同 token 编号顺序在"空间平滑度"和"相邻编号间距"两个口径下的表现。

    - ``neighbour_did``：每个单元与其 4 个最近空间邻居的 |Δtoken_id| 平均（越小 = 编号在二维空间上越平滑）；
      参考基准是对每个单元用 strip id（沿走向几乎不变）算出来的值。
    - ``path_km``：按编号顺序相邻两个 token 质心的距离（RoPE 只看到序号，所以这个越小越好）。
    """
    from scipy.spatial import cKDTree

    centers = fault_centers_km(fault_path, n_elements)
    tree = cKDTree(centers)
    _, nbr = tree.query(centers, k=5)
    out = {}
    for order in orders:
        lay = build_layout(fault_path, n_elements, patch_size, jump_km, order=order, pairing=pairing)
        eid = element_token_id(lay)
        did = float(np.mean(np.abs(eid[nbr[:, 1:]] - eid[:, None])))
        c = lay["token_centroid"]
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)
        out[order] = {
            "neighbour_did": did,
            "path_median_km": float(np.median(d)),
            "path_max_km": float(d.max()),
            "path_gt50": int((d > 50).sum()),
            "layout": lay,
        }
    strips = strip_ids(centers, jump_km)
    out["_strip_id_reference"] = {"neighbour_did": float(np.mean(np.abs(strips[nbr[:, 1:]] - strips[:, None])))}
    return out


def summarize(layout, centers=None, patch_size=None):
    """把布局的关键指标算成一段可打印/可画图的统计（用于验证和对��）。"""
    patch_size = int(patch_size or layout["patch_size"])
    ste = layout["slot_to_element"]
    strips = layout["strip_of_element"]
    n_tok = layout["n_tokens"]
    tok_slots = ste.reshape(n_tok, patch_size)
    tok_strip = strips[tok_slots[:, 0]]
    cross = (strips[tok_slots] != tok_strip[:, None]).any(axis=1)
    out = {
        "n_tokens": n_tok,
        "n_slots": layout["n_slots"],
        "n_elements": int(strips.size),
        "n_strips": layout["n_strips"],
        "n_pad": layout["n_pad"],
        "tokens_crossing_strip": int(cross.sum()),
    }
    if centers is not None:
        d = np.linalg.norm(centers[tok_slots[:, :, None]] - centers[tok_slots[:, None, :]], axis=-1)
        intra = d.reshape(n_tok, -1).max(axis=1)
        out.update({
            "intra_max_km_median": float(np.median(intra)),
            "intra_max_km_p99": float(np.percentile(intra, 99)),
            "intra_max_km_max": float(intra.max()),
            "tokens_over_30km": int((intra > 30).sum()),
            "tok_slots": tok_slots,
            "tok_strip": tok_strip,
            "crossing": cross,
            "intra_km": intra,
        })
    return out
