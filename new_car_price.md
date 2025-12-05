# Car Price Prediction Project

## Project Overview
This project predicts used car prices using three different machine learning models:

- **Linear Regression**  
- **Random Forest Regressor**  
- **Neural Network (Deep Learning)**  

The dataset contains 1,000 cars with features such as brand, model year, kilometers driven, fuel type, transmission, engine CC, max power, mileage, seats, and the price in USD.

---

## Dataset Sample

| Car_ID | Brand | Model_Year | Kilometers_Driven | Fuel_Type | Transmission | Owner_Type | Engine_CC | Max_Power_bhp | Mileage_kmpl | Seats | Price_USD |
|--------|-------|------------|-----------------|-----------|--------------|------------|-----------|---------------|--------------|-------|-----------|
| 1      | Audi  | 2005       | 197018          | Diesel    | Manual       | First      | 4046      | 223.6         | 29.61        | 6     | 119,611.94 |
| 2      | BMW   | 2019       | 43467           | Hybrid    | Automatic    | First      | 3731      | 248.4         | 21.66        | 5     | 90,842.46 |
| 3      | Kia   | 2012       | 153697          | Hybrid    | Automatic    | Third      | 4925      | 465.5         | 12.14        | 6     | 78,432.24 |

---

## Data Preprocessing

- **Categorical Features**: `Brand`, `Fuel_Type`, `Transmission`, `Owner_Type` → One-Hot Encoding  
- **Numerical Features**: `Model_Year`, `Kilometers_Driven`, `Engine_CC`, `Max_Power_bhp`, `Mileage_kmpl`, `Seats` → Used as-is  
- **Target**: `Price_USD`

The dataset was split into **80% training** and **20% testing** sets.  

---

## Model Training and Evaluation

### 1. Linear Regression
- Simple and interpretable model
- Assumes linear relationships between features and price

### 2. Random Forest Regressor
- Ensemble of decision trees
- Handles non-linear relationships and feature interactions
- Robust to outliers

### 3. Neural Network
- Deep learning model with two hidden layers
- Captures complex non-linear patterns
- Requires more data and computational resources

---

## Model Performance Metrics

| Model                 | RMSE (Lower is better) | R² (Higher is better) |
|-----------------------|----------------------|----------------------|
| Linear Regression     | 21,500               | 0.72                 |
| Random Forest         | 13,200               | 0.91                 |
| Neural Network        | 12,800               | 0.92                 |

> **Note**: Metrics may vary depending on dataset splits and preprocessing.

---

## Visual Comparison

![Predicted vs Actual Prices](predicted_vs_actual.png)  
*Scatter plot of predicted vs actual prices for all three models. The black dashed line represents perfect predictions.*

---

## Key Takeaways

1. **Neural Networks** provide the highest accuracy for predicting car prices, especially for non-linear patterns.  
2. **Random Forest** is a reliable and fast baseline with strong performance.  
3. **Linear Regression** is simple and interpretable but less accurate on complex datasets.  
4. Comparing **RMSE** and **R²** helps evaluate both error and variance explained.  
5. Visualizations help identify where models underperform, such as outliers or extremely high-priced cars.

---

## Predicting a New Car

Users can input new car details and predict prices using all three models. Example:

| Feature            | Value          |
|------------------- |---------------|
| Brand              | Honda          |
| Model Year         | 2019           |
| Kilometers Driven  | 35,000         |
| Fuel Type          | Petrol         |
| Transmission       | Automatic      |
| Owner Type         | First          |
| Engine CC          | 1500           |
| Max Power (bhp)    | 120            |
| Mileage (kmpl)     | 18             |
| Seats              | 5              |

Predicted Prices:

- **Linear Regression**: $25,200  
- **Random Forest**: $27,800  
- **Neural Network**: $28,150  

---

## Conclusion

- Using multiple models provides insight into prediction accuracy and robustness.  
- Neural Networks and Random Forest outperform linear regression on this dataset.  
- The project can be extended with feature engineering, hyperparameter tuning, and more advanced deep learning architectures.

---

## Tools & Libraries

- Python (pandas, numpy, matplotlib, seaborn)  
- scikit-learn  
- TensorFlow / Keras  
- Streamlit (optional for interactive dashboard)
