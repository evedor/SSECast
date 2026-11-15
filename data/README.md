# data — not distributed with this repository

The slip-potency fields used in this study are derived from geodetic inversions and
are **not** included in this repository. The training code expects the following
layout; place the fields here (or point `region_configs.<region>` in the YAML at
another location).

```text
data/<region>/
├── slip_potency_smooth/
│   ├── train/{slip, slip_strike, slip_dip}
│   ├── eval/{slip, slip_strike, slip_dip}
│   └── test/{slip, slip_strike, slip_dip}
├── fault/{lat, lon, llh, dip, strike, area, height}
└── norm_value/{mean.npy, std.npy}
```

* `slip`, `slip_strike`, `slip_dip` are whitespace-delimited daily arrays with shape
  `time × fault element` (the three slip-potency components).
* `fault/lat` and `fault/lon` hold three vertex coordinates per triangular element;
  the remaining files carry the corresponding dip, strike, area and depth.
* `norm_value/{mean.npy, std.npy}` must be computed from the **training split only**
  (per channel); they are not distributed here because they are derived from the
  same inversion products.

## How to reconstruct the inputs

1. **Fault discretisation** — discretise the Slab2 interface (Hayes, 2018) into
   triangular elements of ~15 km edge length, matching the element ordering used by
   the slip inversion.
2. **Slip-potency fields** — daily slip-potency inversions are produced with the
   near-real-time SSE workflow
   (<https://near-real-time-sse.esc.cam.ac.uk/cascadia/>) and the preprocessing code
   in <https://github.com/Geolandi/sse_postprocessing> (Julia) or
   <https://github.com/Geolandi/sse_postprocessing_matlab> (MATLAB; the filtering step
   needs a separate implementation).
3. **Normalisation** — compute `mean.npy` / `std.npy` from the training split, per
   channel, and apply them as the code does (z-score).
