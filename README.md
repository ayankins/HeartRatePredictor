# Heart Rate Prediction Model

This repository contains a machine learning model for predicting heart rates based on various physiological and environmental factors. The model leverages logistic regression and random forest algorithms to achieve accurate predictions.

## Features

- Utilizes Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn for data manipulation and visualization.
- Implements Scikit-learn (sklearn) for machine learning tasks including:
  - Data preprocessing with `train_test_split`
  - Model training using `LinearRegression` and `RandomForestRegressor`
  - Evaluation metrics including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

