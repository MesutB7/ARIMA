#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf

# Fetch the data from Yahoo Finance
ticker = 'MSFT'
data = yf.download(ticker, start='2010-01-01', end='2024-07-31', interval='1mo')

# Keep only the 'Close' column
data = data['Close']

# Differencing to make the series stationary
data_diff = data.diff().dropna()

# ACF and PACF plots
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_diff, ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (ACF)')

plt.subplot(122)
plot_pacf(data_diff, ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Model Identification: Using AIC/BIC to select the best ARIMA model
best_aic = np.inf
best_order = None
best_model = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            try:
                model = ARIMA(data, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
            except Exception as e:
                print(f"ARIMA({p},{d},{q}) failed to fit: {e}")
                continue

if best_model is not None:
    print(f'Best ARIMA model: {best_order} with AIC: {best_aic}')
else:
    print("No suitable ARIMA model found.")

# Display statistical results of the best ARIMA model
if best_model is not None:
    print(best_model.summary())

# Plot AR and MA roots in the unit circle
if best_model is not None and best_order[0] > 0:  # Only plot if AR or MA parameters exist
    ar_params = np.r_[1, -best_model.arparams] if best_order[0] > 0 else [1]
    ma_params = np.r_[1, best_model.maparams] if best_order[2] > 0 else [1]

    ar_roots = np.roots(ar_params)
    ma_roots = np.roots(ma_params)

    plt.figure(figsize=(8, 8))
    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    plt.gca().add_patch(unit_circle)

    plt.scatter(ar_roots.real, ar_roots.imag, color='red', label='AR Roots')
    plt.scatter(ma_roots.real, ma_roots.imag, color='green', label='MA Roots')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title('AR and MA Roots in the Unit Circle')
    plt.legend()
    plt.show()

# Unit Root Graph (ADF Test)
adf_test = sm.tsa.adfuller(data_diff)
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')

# Residuals Plot
if best_model is not None:
    residuals = pd.DataFrame(best_model.resid)
    residuals.plot(title="Residuals")
    plt.show()

    # Actual vs Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Actual')
    plt.plot(best_model.fittedvalues, color='red', label='Fitted')
    plt.legend()
    plt.title('Actual vs Fitted')
    plt.show()

    # Forecasting until the end of 2025
    forecast_steps = 17  # Number of months from August 2024 to December 2025
    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='M')[1:]

    # Plot Actual vs Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Actual')
    plt.plot(best_model.fittedvalues, color='red', label='Fitted')
    plt.plot(forecast_index, forecast.predicted_mean, color='green', label='Forecast')
    plt.fill_between(forecast_index, forecast.conf_int()[:, 0], forecast.conf_int()[:, 1], color='green', alpha=0.2)
    plt.legend()
    plt.title('Actual vs Fitted and Forecasted Values')
    plt.show()

else:
    print("Skipping residual and forecast plots since no model was fitted.")


# In[ ]:




