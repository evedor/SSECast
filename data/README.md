# data

## What is included

The **fault geometry** (`data/<region>/fault/`) is distributed with this repository:

```text
data/<region>/fault/{lat, lon, llh, dip, strike, area, height}
```

`lat` and `lon` hold the three vertex coordinates of each triangular element (one row
per element); `dip`, `strike`, `area` and `height` carry the corresponding element
attributes. Element ordering is the ordering used by the slip inversions: the code
relies on it, so do not re-order the rows.

## What is not included

The **slip-potency fields** and the **normalisation statistics** are not distributed:

```text
data/<region>/slip_potency_smooth/{train,eval,test}/{slip, slip_strike, slip_dip}
data/<region>/norm_value/{mean.npy, std.npy}
```

They are derived from geodetic slip inversions and are available from the
corresponding author on reasonable request, or can be regenerated with the sources
below.

## Expected layout

```text
data/<region>/
├── slip_potency_smooth/
│   ├── train/{slip, slip_strike, slip_dip}
│   ├── eval/{slip, slip_strike, slip_dip}
│   └── test/{slip, slip_strike, slip_dip}
├── fault/{lat, lon, llh, dip, strike, area, height}   # included
└── norm_value/{mean.npy, std.npy}
```

* `slip`, `slip_strike`, `slip_dip` are whitespace-delimited daily arrays of shape
  `time × fault element` (the three slip-potency components), in the element order of
  `fault/lat`.
* `norm_value/{mean.npy, std.npy}` must be computed from the **training split only**
  (per channel).

## How to reconstruct the missing inputs

1. **Slip-potency fields** — daily slip-potency inversions are produced with the
   near-real-time SSE workflow
   (<https://near-real-time-sse.esc.cam.ac.uk/cascadia/>) and the preprocessing code
   in <https://github.com/Geolandi/sse_postprocessing> (Julia) or
   <https://github.com/Geolandi/sse_postprocessing_matlab> (MATLAB; the filtering step
   needs a separate implementation).
2. **Normalisation** — compute `mean.npy` / `std.npy` from the training split, per
   channel, and apply them as the code does (z-score).
3. **Fault geometry for a new region** — discretise the Slab2 interface (Hayes, 2018)
   into triangular elements of ~15 km edge length and keep the element ordering used
   by the inversion.

Everything below `data/` is ignored by Git except `data/README.md` and
`data/<region>/fault/` (see `.gitignore`).
