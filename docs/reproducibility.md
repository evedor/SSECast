# Reproducibility notes

SSECast uses two consecutive normalized source fields as input and directly forecasts a specified future horizon. Each source field contains cumulative slip potency, potency along strike and potency along dip. Region-specific models are trained independently.

The canonical multi-horizon training objective combines normalized cumulative-state error with normalized daily-increment error. The upper quartile of the absolute observed slip-potency increment receives additional spatial weight. The mixture-of-experts routing loss is added with its configurable coefficient.

The independent test script reports normalized root-mean-square error and anomaly correlation coefficient for slip potency, slip potency along strike and slip potency along dip at every forecast lead time. No test examples are used for normalization, model selection or optimization.
