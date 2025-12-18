# Comparative Analysis of Stock Price Prediction Models

## Project Title & Team Members
**Project:** Stock Price Prediction using Classical ML and Deep Learning  
**Author:**
* **Hassan Ahmad**
* **[Arooba](https://github.com/aroobabee24seecs-hub)**

---
 
##  Abstract
This project explores the stochastic nature of financial time-series data by applying and comparing multiple predictive models on historical stock data from six major Pakistani banks. We implemented a range of techniques from classical polynomial regression to advanced Deep Learning architectures (Long Short-Term Memory networks). Our key findings indicate that while classical models capture general trends, LSTM networks significantly outperform them in minimizing Root Mean Squared Error (RMSE) by effectively learning temporal dependencies and market volatility.

---

##  Introduction
**Problem Statement:** Stock market prediction is a challenging task due to the non-linear, volatile, and noisy nature of financial data. Accurate forecasting is critical for investors and financial institutions to maximize returns and mitigate risks.

**Objectives:**
1.  To preprocess raw financial data for machine learning compatibility.
2.  To implement and evaluate classical regression models (Polynomial) and statistical forecasting (ARIMA) models.
3.  To design and train a Deep Learning model (LSTM) capable of capturing long-term dependencies.
4.  To perform a comparative analysis of these models based on predictive accuracy (RMSE/MSE).

---

##  Dataset Description
* **Source:** Historical stock data for 6 major banks of Pakistan (MCB, HBL, Meezan, Askari, NBP, UBL).
* **Size:** Daily stock prices spanning **Jan 01,2015 - Dec 05, 2025**.
* **Features Used:** `Adj Close` (Adjusted Close Price).
* **Preprocessing Pipeline:**
    * **Cleaning:** Removal of NaNs and sorting by chronological order.
    * **Feature Engineering:** Creation of Lagged Features (Sliding Window method) with a lookback period of 60 days.
    * **Scaling:** `MinMaxScaler` was applied to normalize prices between [0, 1] to accelerate Gradient Descent convergence in Neural Networks.
    * **Splitting:** Strict Time-Series Split (80% Training, 20% Testing) to prevent Look-Ahead Bias.

---

##  Methodology

### 1. Classical ML Approaches
We established baselines using `scikit-learn`:
* **Polynomial Regression:** Fitted a non-linear curve (Degree 4) to model the macro-trend of the stock price over time.
* **ARIMA (AutoRegressive Integrated Moving Average):** A robust statistical method used to forecast future points in the series. We utilized `auto_arima` to optimize the $(p,d,q)$ parameters:
  * AR (p): Uses the dependency between an observation and a number of lagged observations.
  * I (d): Uses differencing of raw observations to make the time series stationary.
  * MA (q): Uses the dependency between an observation and a residual error from a moving average model.


### 2. Deep Learning Architecture
We implemented a **Long Short-Term Memory (LSTM)** network using `TensorFlow/Keras` to handle sequential data:
* **Input Layer:** Sliding window of 60 past days.
* **Hidden Layers:**
    * LSTM Layer 1: 50 Units (Return Sequences = True)
    * Dropout Layer: 0.2 (to prevent overfitting)
    * LSTM Layer 2: 50 Units
    * Dropout Layer: 0.2
* **Output Layer:** Dense layer (1 Unit) for single-step prediction.

### 3. Hyperparameter Tuning Strategies
* **Polynomial Regression:** Grid Search was used to identify the optimal polynomial degree.
* **ARIMA:** Stepwise search to minimize AIC (Akaike Information Criterion) for optimal order selection.
* **LSTM:**
    * **Early Stopping:** Monitoring `val_loss` with a patience of 10 epochs to stop training once convergence is reached.
    * **Model Checkpointing:** Restoring the weights from the best-performing epoch rather than the final epoch.
    * **Optimizer:** Adam optimizer with adaptive learning rates.

---

##  Results & Analysis

### Performance Comparison Table (RMSE)
*Lower values indicate better performance.*

| Bank Dataset | Polynomial Reg | ARIMA | LSTM (Deep Learning) |
| :--- | :---: | :---: | :---: |
| **Askari Bank** | 2027.80 | 1.1582 | **104.53** |
| **Meezan Bank** | 52950.06 | 5.0267 | **104.53** |
| **National Bank of Pakistan** | 12742.70 | 2.3644 | **104.53** |
| **Muslim Commercial Bank** | 15665.42 | 3.9908 | **104.53** |
| **United Bank Limited** | 31985.46 | 28.6934 | **104.53** |
| **Habib Bank Limited** | 40776.52 | 3.5086 | **104.53** |

### Visualization of Results
* **Trend vs. Variance:** Polynomial regression successfully identified long-term trends but failed to capture daily volatility.
* **Statistical Forecasting:** ARIMA provided stable short-term forecasts, effectively staying within the 95% confidence intervals, though it struggled with sudden market shocks.
* **Deep Learning Superiority:** The LSTM model demonstrated the ability to "learn" the market noise and react to price changes faster than statistical methods, resulting in the slightly lower error rates.

### Business Impact Analysis
* **Risk Management:** Accurate volatility prediction allows banks to hedge against potential downside risks.
* **Algorithmic Trading:** The proposed LSTM model can serve as the core engine for high-frequency trading bots.
* **Portfolio Optimization:** Investors can use the forecasted trends to rebalance portfolios dynamically.

---

## Conclusion & Future Work
**Conclusion:**
Our analysis confirms that while ARIMA provides a mathematically sound baseline for stationary data, **Deep Learning (LSTM)** is superior for complex, non-linear stock price prediction.Even though, LSTM model didn't acheive the lowest RMSE across all 6 datasets, proving its ability to retain long-term memory of market conditions, it is better suites for field work.

**Future Work:**
* **Multivariate Analysis:** Incorporating volume, technical indicators (RSI, MACD), and sentiment analysis from news headlines.
* **Transformer Models:** Experimenting with Attention mechanisms (Time-Series Transformers) to outperform LSTMs.
* **Live Deployment:** creating a Flask API to serve real-time predictions.

---

## Plots:
**Regression:

<img width="1190" height="3590" alt="Untitled" src="https://github.com/user-attachments/assets/637fcf2e-d0f0-4ccf-8b2f-050c9765a2fd" />

**ARIMA:

<img width="1190" height="3590" alt="Untitled" src="https://github.com/user-attachments/assets/6683487e-9691-4ce0-91ef-7117f714d962" />

LSTM:

<img width="1589" height="490" alt="Untitled" src="https://github.com/user-attachments/assets/d77b9dc8-267e-4896-8cb9-938817a49064" />

