# Introduction

In this analysis, I embarked on an exploratory journey through a dataset encapsulating various vehicle characteristics from the 1970s and 1980s, with the objective of unveiling factors that significantly influence a vehicle's fuel efficiency, measured in miles per gallon (mpg). My goal was not only to identify these factors but also to develop a robust model that could predict fuel efficiency based on vehicle attributes.

# Data Overview

The dataset, 'auto-mpg.csv', contains 398 records and 9 fields, including 'mpg' (the target variable), 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', and 'car name'.

Initial data exploration revealed a mix of numerical and categorical variables, with some missing values in 'horsepower'. The 'origin' variable, initially numeric, was converted to categorical to accurately represent different geographical regions.

## Initial Data Handling

Importing Libraries and Loading Data
To commence the analysis, I imported necessary Python libraries for data manipulation, visualization, statistical modeling, and machine learning. Following this, I loaded the 'auto-mpg.csv' dataset into a pandas DataFrame for inspection and processing. An initial peek at the data using data.head() allowed me to familiarize myself with the various attributes available, including 'cylinders', 'displacement', 'horsepower', and others. The preliminary structural overview obtained through data.info() revealed a blend of numerical and categorical data types, and it also flagged potential inconsistencies, particularly within the 'horsepower' attribute, which appeared incorrectly as an object type due to non-numeric entries.

```
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae
import statsmodels.api as sm
import numpy as np

# Loading the dataset
data = pd.read_csv("auto-mpg.csv")
data.head()
```

Preliminary Data Exploration
An initial examination using data.info() revealed a mix of data types and highlighted potential data integrity issues, particularly with the 'horsepower' feature. Additionally, the 'origin' feature was numerically coded, necessitating conversion to a categorical type to properly reflect its meaning.

```

# Checking the basic information and structure of the dataset
data.info()
```

Image

## Data Cleaning

The data cleaning process involved:

1. Converting 'origin' from numerical to categorical.
2. Addressing missing values in 'horsepower' by replacing "?" with NaN and imputing these with the column's mean.

The conversion of 'origin' to a categorical data type is an important data cleaning step, as it correctly represents the nature of this variable. Additionally, the 'horsepower' column is initially an object type due to the presence of "?" characters, indicating missing or malformed data. The code replaces these with NaN and then imputes missing 'horsepower' values with the column's mean.

```
# Converting 'origin' from a numerical to a categorical feature, as it represents different countries
data.origin = data.origin.astype("category")
data.info()
```

Image

```
# Descriptive statistics for numerical features
data.describe()
```

Image

```

# Unique values in 'horsepower' column, identifying potential data issues
data.horsepower.unique()
```

Image

```
# Querying rows where 'horsepower' is missing or marked as "?"
data.query('horsepower == "?"')
```

Image

```
# Mean imputation of missing values in 'horsepower' column
data.horsepower = pd.to_numeric(data.horsepower, errors="coerce")
data.horsepower = data.horsepower.fillna(data.horsepower.mean())
data.info()
```

Image

```
data.describe()
```

Image

Interpretation: These step ensures that 'horsepower' can be utilized in numerical analyses and models. Imputing missing values helps maintain the size of the dataset, although it's important to consider whether mean imputation is the best strategy based on the distribution and importance of 'horsepower'.

# Exploratory Data Analysis (EDA)

## Understanding the Target Variable

I conducted an extensive EDA to understand the underlying patterns within the data. EDA was conducted to understand the distribution of 'mpg' and the relationships between variables:

1. **Histogram of 'mpg':** Revealed a roughly normal distribution with some skewness.

```
# Distribution of the 'mpg' (miles per gallon) target variable
sns.histplot(data.mpg)
```

Image

2. **Pairwise relationships:** Scatter plots and correlation analyses showed varying degrees of association between 'mpg' and other variables.

```# Pairwise relationships between features
sns.pairplot(data, corner=True)
```

Image

3. **Bar plot of 'mpg' by 'origin':** Indicated differences in fuel efficiency across regions.

```
# Relationship between car origin and mpg
sns.barplot(x='origin', y='mpg', data=data)
```

Image

4. **Correlation heatmap:** Helped reveal significant negative associations between 'mpg' and features like 'weight', and positive associations with 'model year'.

```
# Heatmap showing correlations between features, informative for feature selection
sns.heatmap(
    data.corr(numeric_only=True),
    vmin=-1,
    vmax=1,
    cmap='coolwarm',
    annot=True,
)
```

Image

**Interpretation:** These visualizations help identify trends, outliers, and relationships within the data. For example, the 'mpg' histogram and scatter plots can reveal the distribution of fuel efficiency across vehicles and its relationships with other variables. The correlation heatmap aids in identifying potential predictors for regression models by highlighting strong positive or negative correlations with 'mpg'.

# Feature Engineering and Model Preparation

## Enhancing the Dataset

Post-EDA, I decided to introduce a polynomial feature ('weight^2') to explore non-linear relationships. Additionally, 'origin' was one-hot encoded to include as a predictive factor, resulting in a more nuanced model that considers geographical influences.

```
# Adding a polynomial feature for 'weight' and removing the 'car name' feature
data_model = data.assign(weight2=data.weight**2).drop("car name", axis=1)
# Encoding categorical 'origin' feature as dummy variables
data_model = pd.get_dummies(data_model, drop_first=True, dtype=int)  # Specify dtype=int here
data_model.head()
```

Image

**Interpretation:** By extending the feature set and using cross-validation, you can evaluate the model's ability to generalize to unseen data and select the most relevant predictors. The residual analysis, including the residual plot and Q-Q plot, is crucial for verifying that the residuals are normally distributed and have constant variance, ensuring that the linear model is appropriate for the data.

## Preparing for Regression Analysis

Features were selected and prepared for modeling. A constant was added to account for the intercept in the regression model.

```
# Defining features for the regression model
features = [
  "weight",
  "weight2",
  "model year",
  "origin_2",
  "origin_3",
]

# Adding a constant to the model for the intercept
X = sm.add_constant(data_model[features])
y = data_model["mpg"]

# Splitting the dataset into training and testing sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
```

# Model Evaluation and Diagnostics

## Implementing OLS Regression

In my endeavor to understand how vehicle attributes like weight and model year impact miles per gallon (mpg), I turned to Ordinary Least Squares (OLS) Regression. This method aims to identify the linear relationship between independent variables and the dependent variable by minimizing the squared differences between observed and predicted values.

## Model Evaluation Process

Cross-Validation: Understanding the importance of validating the model’s predictive power and ensuring its applicability across different data samples, I utilized K-Fold cross-validation. This method splits the dataset into multiple segments, enabling a comprehensive assessment of the model’s performance and ensuring robustness.

```
# Setting up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2023)

# Cross-Validation for Model Evaluation
cv_lm_r2s = []  # list to store R2 scores for each fold
cv_lm_maes = []  # list to store MAE for each fold

# Loop through folds, fit model, and evaluate performance
for train_ind, val_ind in kf.split(X,y):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
    X_val, y_val = X.iloc[val_ind], y.iloc[val_ind]

    # Fitting the model
    model = sm.OLS(y_train, X_train).fit()

    # Storing R2 and MAE for each validation fold
    cv_lm_r2s.append(r2(y_val, model.predict(X_val)))
    cv_lm_maes.append(mae(y_val, model.predict(X_val)))

# Printing cross-validation R2 and MAE scores
print("All Validation R2: ", [round(x, 3) for x in cv_lm_r2s])
print(f"Cross Val R2s: {round(np.mean(cv_lm_r2s), 3)} +- {round(np.std(cv_lm_r2s), 3)}")
print("All Validation MAE: ", [round(x, 3) for x in cv_lm_maes])
print(f"Cross Val MAEs: {round(np.mean(cv_lm_maes), 3)} +- {round(np.std(cv_lm_maes), 3)}")
```

```
All Validation R2:  [0.881, 0.866, 0.835, 0.84, 0.837]
Cross Val R2s: 0.852 +- 0.019
All Validation MAE:  [1.879, 2.244, 2.396, 2.209, 2.394]
Cross Val MAEs: 2.224 +- 0.189
```

## Diagnostic Checks: Residual Analysis

The residual analysis was crucial in validating the assumptions of linear regression. The patterns in the residual plot and the Q-Q plot indicated that while there were deviations, they were not severe enough to undermine the model's validity.

```
def residual_analysis_plots(model):
    import scipy.stats as stats

    predictions = model.predict()
    residuals = model.resid

    # Creating plots: scatter plot for residuals and Q-Q plot for normality check
    fig, ax = plt.subplots(1,2, sharey="all", figsize=(10,6))

    sns.scatterplot(x=predictions, y=residuals, ax=ax[0])
    ax[0].set_title("Residual Plot")
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("Residuals")

    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q Plot")

    # Applying residual analysis on the fitted model


residual_analysis_plots(model)
```

Image

```
# Summary of the regression model results
model.summary()
```

Image

## Reflecting on My Findings

From applying OLS regression within this structured cross-validation framework, I unraveled significant insights. I learned how specific vehicle attributes can sway fuel efficiency, which in turn informs vehicle design and policy decisions aimed at improving mpg.

The nuanced understanding gained from this regression analysis not only shed light on the quantitative impact of each variable but also painted a clearer picture of the automotive industry's challenges and opportunities in enhancing fuel efficiency.

# Final Model Evaluation

With the final OLS model, I moved forward to assess its performance on the test data. The results echoed the findings from the cross-validation, cementing my confidence in the model's capabilities.

```
# Refitting model with the entire training set
model = sm.OLS(y, X).fit()
```

**Model Refitting:** After conducting cross-validation and ensuring that the model was adequately tuned, I refitted the Ordinary Least Squares (OLS) model using the entire dataset. This step is vital to leverage all available data for the final model, which could enhance its predictive accuracy and stability.

```
# Evaluating performance on the test set
print(f"Test R2: {r2(y_test, model.predict(X_test))}")
print(f"Test MAE: {mae(y_test, model.predict(X_test))}")
```

```
Test R2: 0.8087480897721645
Test MAE: 2.2659313639801715
```

**Performance Evaluation on Test Data:** I evaluated the refitted model's performance on the separate test dataset. This phase aimed to test the model's generalizability. The R² (Coefficient of Determination) and MAE (Mean Absolute Error) were my focal metrics, providing insights into the model's explanatory power and predictive accuracy, respectively.

**Comparative Analysis with Validation Set:** In addition to the test evaluation, I revisited the performance metrics obtained from the validation phase during cross-validation. This comparison is crucial for assessing the model's consistency and reliability across different data subsets.

**Interpretation:** The test set results offered definitive insights into the model's performance in a real-world scenario. A high R² in conjunction with a low MAE reaffirmed the model's effectiveness in predicting vehicle fuel efficiency. These findings underscore the model's practical applicability and underscore its potential utility in automotive industry and policy-making.

The comparative analysis between the test metrics and validation phase outcomes provided an added layer of validation, ensuring that the model's predictive performance is robust and not merely a result of overfitting to the training data.

```
# Final model summary
model.summary()
```

Image

The final model summary reveals detailed statistics, including the coefficients, standard errors, and confidence intervals for each predictor. This comprehensive overview allows for an in-depth interpretation of how each feature influences vehicle fuel efficiency.

## Ridge Regression Analysis

Acknowledging the potential for multicollinearity and overfitting, I incorporated Ridge regression into my analysis. The regularization process, optimized through cross-validation, improved the model's generalizability, evidenced by enhanced R² and MAE metrics on the test data compared to the OLS model.

```
# Standardizing the features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
scaler = StandardScaler()
X_mat = scaler.fit_transform(X.values)
X_test_mat = scaler.transform(X_test.values)
```

**Standardizing Features:** For the Ridge regression model, I standardized the features to ensure they're on the same scale. This is crucial because regularization penalizes larger coefficients; hence, features need to be standardized to avoid biasing the model against features with larger magnitudes.

```
# Setting up Ridge regression with cross-validation
n_alphas = 200
alphas = np.logspace(-5, 2, n_alphas)  # Defining a range of alpha values for Ridge
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X_mat, y)
```

**Ridge Regression with Cross-validation:** I implemented Ridge regression, which introduces a regularization term to the cost function, to prevent overfitting. By using RidgeCV with cross-validation, I automatically selected the best regularization strength (alpha) from a range of potential values, optimizing the model's performance and generalizability.

```

# Evaluating Ridge regression model
ridge_r2 = ridge_model.score(X_mat, y)
ridge_mae = mae(y, ridge_model.predict(X_mat))
ridge_alpha = ridge_model.alpha_
print(f"RidgeCV R2: {ridge_r2}")
print(f"RidgeCV MAE: {ridge_mae}")
print(f"RidgeCV Alpha: {ridge_alpha}")
```

**Evaluating Ridge Regression Model:** The performance of the Ridge regression model was evaluated on the standardized training data. The R² (Coefficient of Determination) and MAE (Mean Absolute Error) metrics provided insights into the model's explanatory power and predictive accuracy, respectively, while the chosen alpha value indicated the level of regularization applied.

```
# Evaluating performance on the test set
test_r2_ridge = r2(y_test, ridge_model.predict(X_test_mat))
test_mae_ridge = mae(y_test, ridge_model.predict(X_test_mat))
print(f"Test R2 (Ridge): {test_r2_ridge}")
print(f"Test MAE (Ridge): {test_mae_ridge}")
```

**Interpretation of Ridge Regression Results:** The Ridge regression model's performance metrics on both the training and test datasets offer insights into its effectiveness in predicting vehicle fuel efficiency under regularization constraints. The comparison of R² and MAE between the Ridge and OLS models reveals the impact of regularization on the model's predictive capability and robustness against overfitting. The chosen alpha value reflects the model's balance between complexity and fit, informing the extent of penalty applied to larger coefficients to enhance model generalizability.

In conclusion, the Ridge regression analysis extends our understanding of vehicle fuel efficiency determinants, addressing potential overfitting issues inherent in the standard OLS approach. This robust model underscores the importance of feature scaling and regularization in predictive modeling, offering a valuable tool for enhanced predictive accuracy and model interpretability in the context of automotive data analysis. Future investigations could further explore alternative regularization techniques or hybrid models to refine predictions and uncover deeper insights.

# Conclusions and Future Directions

My journey through this dataset led to several enlightening insights:

Vehicle weight and the model year significantly affect fuel efficiency, corroborating intuitive expectations and historical automotive trends.
Regional variations (represented by 'origin') also play a critical role, likely reflecting different manufacturing standards and environmental regulations.
The application of polynomial features and regularization techniques like Ridge regression were instrumental in refining the model's performance and interpretability.
The journey through this project was an illuminating testament to the power of data-driven insights in understanding complex relationships like those influencing vehicle fuel efficiency. For future explorations, I aim to integrate more advanced modeling techniques, explore the interaction effects between variables, and possibly extend the analysis to more contemporary data, paving the way for a deeper understanding and more accurate predictions of fuel efficiency in the evolving landscape of automotive technology.
