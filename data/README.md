# Data layout

This repository does not distribute GNSS observations, inversion products, tremor catalogues, trained checkpoints or forecast outputs.

For a region named `cascadia`, the configuration expects:

```text
data/
  cascadia/
    slip_potency_smooth/
      train/{slip,slip_strike,slip_dip}
      eval/{slip,slip_strike,slip_dip}
      test/{slip,slip_strike,slip_dip}
    norm_value/{mean.npy,std.npy}
    tremor/{train,eval,test}/tremor_density
    fault/
```

The source fields are whitespace-delimited daily arrays with shape `time × fault element`. Data provenance, access conditions and preprocessing are documented separately with the associated study.
