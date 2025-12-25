Stock Investment Advisor (Streamlit + Plotly + uv + DuckDB)

## 1) Overview

This project is a Python-based **investment advisor dashboard** that:

1. Fetches historical OHLCV data from **free providers** (v1: **Stooq**)
2. Stores/caches data locally in **DuckDB**
3. Applies basic filters/indicators (moving averages, volatility, RSI)
4. Produces **30-day forecasts** using:

   * Moving Average baseline
   * RandomForest (scikit-learn) **multi-horizon direct forecasting of returns**
5. Generates a **Buy / Hold / Sell** suggestion using a **transparent, return-based scoring** system
6. Presents everything in an interactive **Streamlit + Plotly** UI

**Disclaimer:** This is a research/educational tool. It does not provide financial advice.

---

## 2) Tech stack

* **Python 3.11+**
* **uv** for dependency management and virtual environments
* **Streamlit** UI
* **Plotly** charts
* **pandas** data processing
* **scikit-learn** RandomForest modeling
* **DuckDB** for local caching/persistence

---

## 3) Quickstart

### 3.1 Prerequisites

* Install `uv` (recommended):

  * macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  * Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 3.2 Setup

From the project root:

```bash
uv venv
uv sync
```

### 3.3 Run the Streamlit app

```bash
uv run streamlit run app.py
```

### 3.4 Run tests (optional)

```bash
uv run pytest -q
```

---

## 4) Project layout (minimal, modular)

```
.
├── app.py
├── pyproject.toml
├── README.md
├── data/
│   └── cache.duckdb
├── src/
│   ├── config.py
│   ├── storage_duckdb.py
│   ├── providers/
│   │   ├── base.py
│   │   └── stooq.py
│   ├── indicators.py
│   ├── features.py
│   ├── models/
│   │   ├── base.py
│   │   ├── moving_average.py
│   │   └── random_forest.py
│   ├── backtest.py
│   ├── advisor.py
│   └── plotting.py
└── tests/
   └── test_smoke.py
```

---

## 5) Data: providers + canonical schema

### 5.1 Canonical data format

All providers must return a dataframe normalized to:

* `date` (datetime/date)
* `open`, `high`, `low`, `close`, `volume` (float)
* `symbol` (string)
* `provider` (string)

### 5.2 Provider rules

* v1 implements **StooqProvider**
* Provider interface: `get_ohlcv(symbol, start, end, interval="1d") -> pd.DataFrame`
* If a provider cannot fetch the requested asset (e.g., crypto), the UI must show a clear message and skip forecasting until data is available.

---

## 6) Persistence: DuckDB caching

### 6.1 DuckDB file

* Stored at: `data/cache.duckdb`

### 6.2 Table schema

Single table:

```sql
CREATE TABLE IF NOT EXISTS ohlcv (
  symbol TEXT,
  provider TEXT,
  date DATE,
  open DOUBLE,
  high DOUBLE,
  low DOUBLE,
  close DOUBLE,
  volume DOUBLE,
  PRIMARY KEY(symbol, provider, date)
);
```

### 6.3 Storage interface

Implement:

* `upsert_ohlcv(df)`
* `load_ohlcv(symbol, provider, start, end)`
* `latest_date(symbol, provider)`

### 6.4 Cache logic

When user requests data:

1. Check DB for the requested range
2. Only fetch missing dates from provider
3. Upsert fetched data into DuckDB
4. Return complete merged data from DB

---

## 7) Indicators and filters

Indicators should be computed from the canonical OHLCV dataframe.

Must include:

* **SMA**: 4w, 12w, 26w (interpret as 20, 60, 130 trading days)
* **EMA**: same windows
* **Rolling volatility** (e.g., 20-day std of returns)
* **RSI** (14-day recommended)

Implementation constraints:

* No plotting inside indicator functions
* Functions should be pure and testable

---

## 8) Forecasting models

### 8.1 Forecast horizon

Default horizon: **30 trading days**

### 8.2 Baseline: Moving Average forecast

* Compute a moving average of returns (or price), then produce horizon forecasts
* v1 baseline should be return-based for consistency:

  * forecasted return each day = last rolling mean return
  * predicted price path derived by compounding

### 8.3 RandomForest multi-horizon (return-based)

RandomForest forecasts **future returns**, not prices.

#### 8.3.1 Target

* Model predicts **H-step future returns** directly, for H=30 days:

  * `y[t] = [r(t+1), r(t+2), ..., r(t+30)]`
* Use `MultiOutputRegressor(RandomForestRegressor(...))` OR an equivalent multi-output strategy.

#### 8.3.2 Features (no leakage)

For time t, build features only from data ≤ t:

* lagged returns: `r(t), r(t-1), ... r(t-n)`
* lagged close: `close(t), close(t-1), ...`
* rolling stats: rolling mean/std of returns, volatility, RSI
* moving average distances: `close - SMA20`, etc.

#### 8.3.3 Training split

* Time-based split only (no shuffle)
* Example:

  * Train: first 80%
  * Validation: last 20%
* Any backtesting must be walk-forward or at least a strict holdout.

#### 8.3.4 Predicted price path

Convert predicted returns into a price path:

* Let last known price be `P0`
* For predicted returns `r1..r30`:

  * `P1 = P0 * (1 + r1)`
  * `P2 = P1 * (1 + r2)`
  * etc.

---

## 9) Evaluation / backtest

v1 should include a lightweight backtest:

* Backtest window: last **180 days**
* For each evaluation point:

  * train on history up to t
  * predict next 30d
  * evaluate against realized returns

Compute metrics:

* MAE / RMSE / MAPE (on price path or returns; prefer returns)
* Directional accuracy (sign correctness)

Output:

* metrics table in UI
* optional plot overlay of backtest predictions vs actual

---

## 10) Advisor: Buy / Hold / Sell (return-based)

### 10.1 Inputs

Use these signals:

* Expected 30-day return (mean of predicted returns or compounded return)
* Volatility (rolling)
* Trend filter (SMA20 vs SMA60, slope, or MA distance)
* RSI regime (optional)

### 10.2 Decision policy (transparent and stable)

Compute:

1. **Expected compounded return over horizon**
   `R_30 = Π(1 + r_i) - 1`

2. **Risk penalty**
   `risk = rolling_vol_20 * sqrt(30)` (scaled)

3. **Score**
   `score = R_30 - λ * risk`

   * λ defaults to ~0.5 (tunable)

4. **Trend gate**

   * If SMA20 < SMA60, reduce score (bearish regime)

5. **Decision thresholds** (defaults)

* BUY if `score > +0.03` and trend not bearish
* SELL if `score < -0.03`
* HOLD otherwise

6. **Confidence**

* High if |score| > 0.06 and volatility not extreme
* Medium if |score| 0.03–0.06
* Low otherwise

### 10.3 Advisor output

UI must show:

* BUY / HOLD / SELL
* Confidence (Low/Med/High)
* Explanation text listing:

  * expected 30d return
  * volatility
  * trend regime
  * score calculation summary
* Disclaimer text always visible

---

## 11) Streamlit UI requirements

### 11.1 Controls

* Symbol input (text)
* Asset type dropdown: equity / ETF / index / crypto
* Provider dropdown (default Stooq)
* Date range selector
* Horizon selector (default 30; allow override)
* Model selection: MA / RF / Both
* Buttons:

  * Fetch/Update Data
  * Run Forecast

### 11.2 Outputs

1. Plotly chart:

   * Close price
   * SMA/EMA overlays
   * Forecast extension lines (dashed)
   * Optional confidence band later
2. Metrics table
3. Advisor panel (decision + confidence + explanation)
4. Data table preview (optional)

### 11.3 Performance requirements

* Use Streamlit caching for:

  * database reads
  * provider fetches (by parameters)
* Avoid training models unless the user requests it

---

## 12) Running with uv (exact commands)

### Create venv + install

```bash
uv venv
uv sync
```

### Run app

```bash
uv run streamlit run app.py
```

### Run tests

```bash
uv run pytest -q
```

### Add a dependency

```bash
uv add <package>
```

---

## 13) Acceptance checklist

The project is “done” when:

* Entering a symbol and date range fetches OHLCV (stored in DuckDB)
* Indicators render on the Plotly chart
* Forecasting works for MA and RandomForest
* Forecasted returns are converted to a price path and plotted
* Backtest metrics display
* Advisor panel outputs Buy/Hold/Sell + confidence + explanation
* Everything runs via uv + streamlit

---

## 14) Next extensions (optional)

* Add CoinGecko provider for crypto OHLCV
* Add additional models: XGBoost, Prophet, ARIMA
* Add portfolios/watchlists stored in DuckDB
* Add transaction cost assumptions to advisor
* Add regime detection and calibration

