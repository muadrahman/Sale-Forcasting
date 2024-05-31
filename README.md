# Sales Prediction with Random Forest and XGBoost Regressors

## Overview

This project focuses on predicting sales using two powerful machine learning algorithms: Random Forest and XGBoost regressors. Leveraging a comprehensive dataset comprising various features related to items and outlets, the models are trained to accurately forecast sales, providing valuable insights for business decision-making.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Acquisition](#data-acquisition)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Results](#results)
8. [Technologies Used](#technologies-used)
9. [Contact](#contact)
10. [References](#references)
11. [Project Link](#project-link)

## Introduction

Sales prediction is a critical task for businesses, enabling them to anticipate demand, optimize inventory management, and enhance overall operational efficiency. By employing advanced machine learning techniques, this project aims to deliver accurate sales forecasts based on historical data, facilitating informed decision-making and driving business growth.

## Data Acquisition

The dataset is sourced from a CSV file containing comprehensive information about items and outlets. Utilizing the Pandas library, the dataset is loaded into a DataFrame, laying the foundation for further analysis and modeling.

## Data Preprocessing

Prior to model training, the dataset undergoes thorough preprocessing steps to ensure data quality and consistency. Missing values are addressed through appropriate imputation techniques, and categorical features are encoded to facilitate model training.

## Feature Engineering

Feature engineering plays a crucial role in enhancing model performance. New features such as 'Outlet_Years' and 'New_Item_Type' are created to capture additional insights from the data, enriching the predictive capabilities of the models.

## Model Training

The project employs two state-of-the-art regression algorithms: Random Forest and XGBoost. These models are trained on the preprocessed data, leveraging the scikit-learn library for seamless model implementation and training.

### Steps:

1. **Data Splitting:** The dataset is split into training and testing sets.
2. **Random Forest Model:** Initialize and train a Random Forest regressor using the training data.
3. **Hyperparameter Tuning:** Utilize RandomizedSearchCV to search for the best hyperparameters for the Random Forest model.
4. **Model Evaluation:** Evaluate the trained Random Forest model using the R2 score metric.
5. **XGBoost Model:** Initialize and train an XGBoost regressor using the training data.
6. **Model Evaluation:** Evaluate the trained XGBoost model using the R2 score metric.

## Model Evaluation

Model performance is evaluated using the R2 score metric, providing valuable insights into the models' predictive accuracy and effectiveness in capturing the variability in sales data.

## Results

The Random Forest model achieves an impressive R2 score of 0.726, highlighting its robust performance in accurately predicting sales. These results underscore the efficacy of machine learning techniques in delivering actionable insights for businesses.

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost

## Contact

For inquiries or feedback, please feel free to reach out:

- [Email](mailto:mr.muadrahman@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/muadrahman/)

## References

For further reading and exploration, refer to the following documentation:

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## Project Link

To access the project repository and explore the codebase, visit [this link](https://github.com/your-username/sales-prediction).
