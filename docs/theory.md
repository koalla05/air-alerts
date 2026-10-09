# Comprehensive Guide to Time Series Analysis

Time series analysis models data points collected sequentially over time $t \in T$. Unlike standard statistical modeling, time series data violates the Independent and Identically Distributed (i.i.d.) assumption due to serial dependencies across observations.

Data structures fall into two main representations:
* **Regular Time Series**: Sampled at uniform intervals $\Delta t$ (e.g., hourly aggregation, daily sum).
* **Irregular / Point Process Data**: Events occur at arbitrary timestamps $t_1, t_2, \dots, t_n$. Analysis focuses on inter-arrival times $\tau_i = t_i - t_{i-1}$, event counts in time windows, or continuous intensity estimation.

---

## 1. Time Series Decomposition

A time series $y_t$ can be factorized into four primary component processes:

1. **Additive Model**: Used when seasonal/cyclical variations remain constant regardless of the trend magnitude.
   $$y_t = T_t + S_t + C_t + \epsilon_t$$
2. **Multiplicative Model**: Used when the amplitude of variations scales proportionally with the trend.
   $$y_t = T_t \cdot S_t \cdot C_t \cdot \epsilon_t$$

### Components
* **Trend ($T_t$)**: Long-term directional movement (upward, downward, or neutral) over the entire timeframe.
* **Seasonality ($S_t$)**: Patterns repeating at fixed, deterministic calendar frequencies (e.g., daily hour-of-day, weekly day-of-week).
* **Cyclicality ($C_t$)**: Non-periodic, longer-term fluctuations driven by macro-factors without a fixed structural length.
* **Residual / Noise ($\epsilon_t$)**: Irreducible random variation remaining after removing structured components.

### Decomposition Methods
* **Classical Decomposition**: Uses centered moving averages for trend estimation and average ratios for seasonal indexing.
* **STL (Seasonal-Trend Decomposition using Loess)**: Non-parametric method using locally weighted regression. Handles flexible seasonal changes and is robust against outliers.

---

## 2. Stationarity and Transformations

Stationarity requires that the statistical properties of the series remain constant over time.

### Weak (Second-Order) Stationarity
A time series $y_t$ is weakly stationary if it satisfies three criteria:
1. **Constant Mean**: $\mathbb{E}[y_t] = \mu$ for all $t$.
2. **Constant Variance**: $\text{Var}(y_t) = \sigma^2 < \infty$ for all $t$.
3. **Time-Invariant Autocovariance**: $\text{Cov}(y_t, y_{t-k}) = \gamma(k)$, depending strictly on lag distance $k$, not time $t$.

### Strict Stationarity
The joint probability distribution of any sequence $(y_{t_1}, \dots, y_{t_k})$ is identical to the shifted sequence $(y_{t_1+h}, \dots, y_{t_k+h})$ for any time shift $h$.

### Diagnostic Tests
* **Augmented Dickey-Fuller (ADF) Test**: Tests $H_0$: Unit root present (non-stationary) vs. $H_1$: Stationary.
* **KPSS Test**: Tests $H_0$: Trend/Level stationary vs. $H_1$: Unit root present.
* *Dual Testing Strategy*: Confirmed stationarity occurs when ADF rejects $H_0$ and KPSS fails to reject $H_0$.

### Stationary Transformations
* **First-Order Differencing**: $\Delta y_t = y_t - y_{t-1}$ removes linear trends.
* **Seasonal Differencing**: $\Delta_s y_t = y_t - y_{t-s}$ eliminates repeating period-$s$ seasonal effects.
* **Variance Stabilization**: Logarithmic transform $\ln(y_t)$ or Box-Cox transformation $\frac{y_t^\lambda - 1}{\lambda}$ stabilizes changing variance (heteroscedasticity).

---

## 3. Autocorrelation Analysis

Dependency metrics across time lags govern the structural configuration of parametric models.

* **Autocovariance Function**:
  $$\gamma(k) = \mathbb{E}[(y_t - \mu)(y_{t-k} - \mu)]$$
* **Autocorrelation Function (ACF)**:
  $$\rho_k = \frac{\gamma(k)}{\gamma(0)} \in [-1, 1]$$
  Measures linear dependency between $y_t$ and $y_{t-k}$, including intermediate lag influences.
* **Partial Autocorrelation Function (PACF)**:
  Measures direct correlation between $y_t$ and $y_{t-k}$ after partialing out linear correlations of intermediate lags $(y_{t-1}, \dots, y_{t-k+1})$.

| Model Specification | ACF Profile | PACF Profile |
| :--- | :--- | :--- |
| **AR($p$)** | Exponential decay or sinusoidal damping | Cuts off abruptly after lag $p$ |
| **MA($q$)** | Cuts off abruptly after lag $q$ | Exponential decay or sinusoidal damping |
| **ARMA($p, q$)** | Exponential decay after lag $q$ | Exponential decay after lag $p$ |

---

## 4. Parametric Time Series Models

### Autoregressive Model: AR($p$)
Expresses $y_t$ as a linear combination of its $p$ past values:
$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t$$

### Moving Average Model: MA($q$)
Expresses $y_t$ as a linear combination of $q$ past white noise residual shocks:
$$y_t = \mu + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

### Autoregressive Integrated Moving Average: ARIMA($p, d, q$)
Combines AR and MA modeling after applying $d$-th order differencing to achieve stationarity:
$$\left(1 - \sum_{i=1}^p \phi_i L^i\right) (1 - L)^d y_t = c + \left(1 + \sum_{j=1}^q \theta_j L^j\right) \epsilon_t$$
where $L$ is the backshift operator ($L^k y_t = y_{t-k}$).

### Seasonal ARIMA: SARIMA($p, d, q$) $\times$ ($P, D, Q$)$_s$
Extends ARIMA by adding seasonal autoregressive ($P$), differencing ($D$), and moving average ($Q$) terms with seasonal period length $s$.

### Multivariate Frameworks
* **VAR (Vector Autoregression)**: Models dynamic interactions between vector components $\mathbf{Y}_t = [y_{1,t}, y_{2,t}, \dots, y_{m,t}]^T$.
* **ARIMAX / Dynamic Regression**: Adds external covariates $X_t$ directly to the ARIMA mathematical formulation.

---

## 5. Event-Based Time Series & Point Processes

When modeling discrete events rather than continuous metrics, stochastic point processes and structural boundary methods apply:

### 1. Temporal Point Processes
Events are represented as timestamp sequences $T = \{t_1, t_2, \dots, t_n\}$. The process is characterized by its conditional intensity function $\lambda(t \mid \mathcal{H}_t)$, representing the instantaneous event rate at time $t$ given past history $\mathcal{H}_t$:
$$\lambda(t \mid \mathcal{H}_t) = \lim_{\Delta t \to 0} \frac{\mathbb{P}(\text{Event in } [t, t+\Delta t] \mid \mathcal{H}_t)}{\Delta t}$$

### 2. Hawkes Process (Self-Exciting Point Process)
Models cluster-based event patterns where an initial event elevates the probability of subsequent events over a localized time horizon:
$$\lambda(t) = \mu(t) + \sum_{t_i < t} \alpha \, \gamma(t - t_i)$$
where $\mu(t)$ is baseline intensity, $\alpha$ is excitation scale, and $\gamma(\Delta t) = e^{-\beta \Delta t}$ represents exponential memory decay.

### 3. Change-Point Detection
Identifies non-stationary structural shifts in distribution, mean rate, or variance across time bounds:
* **Algorithms**: Pruned Exact Linear Time (PELT), Binary Segmentation, Bayesian Change-Point Analysis (BCPA).

### 4. Duration & Hazard Modeling
Analyzes interval length $D_i = t_{\text{end}, i} - t_{\text{start}, i}$ using survival analysis metrics:
* **Hazard Function $h(t)$**: Instantaneous termination rate given survival up to duration $t$.
* **Formulations**: Weibull parametric regression, Cox Proportional Hazards model.

---

## 6. Machine Learning & Deep Learning Frameworks

* **Feature Engineering**: Lagged target values ($y_{t-1}, y_{t-24}$), rolling stats (window mean, window standard deviation), cyclical encodings ($\sin/\cos$ transformations of time components).
* **Gradient Boosted Decision Trees**: XGBoost, LightGBM, CatBoost using temporally ordered splitting or Purged Group Time-Series Validation.
* **Recurrent Architectures**: LSTM, GRU for temporal sequential dependency tracking.
* **Modern Time Series Deep Learning**:
  * **Temporal Fusion Transformer (TFT)**: Multi-horizon architecture designed for static metadata, past observed inputs, and known future features.
  * **N-BEATS / N-HiTS**: Fully connected neural architectures utilizing basis function expansion for exact trend and seasonal breakdown.
  * **PatchTST**: Channel-independent Transformer architecture using temporal sub-sequence patching.

---

## 7. Educational Resources & Further Reading

### Open-Access Textbooks
* [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) by Rob J. Hyndman and George Athanasopoulos *(Essential textbook for classical decomposition, ARIMA, and evaluation)*.
* [Statistical Modeling: A Fresh Approach](https://dtkaplan.github.io/SM2-bookdown/) *(Introductory probability and regression concepts)*.

### Libraries & Documentation
* [Statsmodels Time Series Analysis (statsmodels.tsa)](https://www.statsmodels.org/stable/tsa.html) *(Python library for ADF/KPSS tests, ARIMA, SARIMAX, VAR)*.
* [Ruptures Documentation](https://nimblelearn.org/ruptures/) *(Python library for offline change-point detection algorithms like PELT)*.
* [Lifelines Documentation](https://lifelines.readthedocs.io/) *(Survival analysis and hazard modeling in Python)*.
* [Darts Documentation](https://unit8co.github.io/darts/) *(Unified Python framework for forecasting models from ARIMA to Transformers)*.
* [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) *(Implementation of Temporal Fusion Transformer, N-BEATS, and DeepAR)*.

### Key Papers
* [Hawkes Processes (1971)](https://doi.org/10.2307/2334319): Alan G. Hawkes, *Spectra of Some Self-Exciting and Mutually Exciting Point Processes*.
* [STL Decomposition (1990)](https://www.w3.org/1999/02/22-rdf-syntax-ns#): R. B. Cleveland et al., *STL: A Seasonal-Trend Decomposition Procedure Based on Loess*.
* [Temporal Fusion Transformer (2021)](https://arxiv.org/abs/1912.09363): Bryan Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
* [PatchTST (2023)](https://arxiv.org/abs/2211.14730): Yuqi Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.