# Reproducibility notes

SSECast uses two consecutive normalized source fields as input and directly forecasts a specified future horizon. Each source field contains cumulative slip potency, potency along strike and potency along dip. Region-specific models are trained independently.

The canonical multi-horizon training objective combines normalized cumulative-state error with normalized daily-increment error. The upper quartile of the absolute observed slip-potency increment receives additional spatial weight. The mixture-of-experts routing loss is added with its configurable coefficient.

The evaluation script reports normalized root-mean-square error and anomaly correlation coefficient for the slip-potency component, alongside causal reference forecasts: persistence, linear extrapolation, local trend, first-order autoregressive increments, empirical recurrence and nearest observed analogues.
