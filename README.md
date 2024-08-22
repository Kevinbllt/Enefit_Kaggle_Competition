# Electricity Consumption Prediction



## Description

This project aims to predict electricity consumption using advanced machine learning techniques. The project is part of a broader context where the integration of prosumers (producer-consumers) into the power grid is becoming increasingly crucial. The goal is to minimize imbalance costs, which occur when energy consumption and production forecasts are inaccurate, leading to overloads or underloads in the grid. This project is a $50k competition available on Kaggle, created by the company ENEFIT, based in Estonia.

Through this project, my personal objective was to gain experience in analyzing energy sector data and developing and optimizing machine learning models to ensure high prediction accuracy. Therefore, this project serves both an educational purpose for me and a demonstration of my data skills on a real-world problem that businesses may encounter.



## Table of content

1. Project structure
2. Models and methodologies
3. Results
4. Areas for improvement
5. Contribution



## 1 - Project structure

- `Train_files/`: datasets used for model training.
- `Test_files/`:  datasets used for testing the best model.
- `enefit/`: default dataset provided at the start of the competition, intended for model deployment. Not used in my project.
- `Backup notebooks/`: backup of notebooks made throughout the project for safety.

- `1. Data Exploration and Analysis/`: exploration of the datasets used in this project. This file is divided into different sections, each analyzing a specific dataset. In each section, I attempt to answer business-related questions by manipulating and analyzing the data.
- `2. Models Exploration/`: data preprocessing (merging tables, cleaning, feature engineering, etc.) as well as the study, testing, and optimization of various models (LGBM, XGBoost, Random Forest, etc.).
- `3. Main Notebook and business study/`: refinement of the code into clear and precise functions. The best model is optimized and its performance is evaluated from a business perspective, specifically by analyzing the gains achieved with this model.



## 2 - Models and Methodologie

### 2.1 - Preprocessing

a. Dataset merging
The first step in my preprocessing involved merging all the data into a single dataset. It was crucial to merge the datasets intelligently, using the best columns and the most appropriate methods (inner, left, etc.).

b. Data cleaning and transformation
- Handling missing values: very few values were missing in the data (<1%). The chosen strategy was to remove rows with missing values.
- Handling duplicate rows: there were no duplicate rows in the data.
- Handling highly correlated columns: to avoid retaining columns that added little value and could introduce noise into my models, I identified and removed columns with a correlation greater than 0.9.
- Handling different column types: during preprocessing, it was important to ensure that date features were in the correct format (%Y-%m-%d %H:%M:%S or %Y-%m-%d). There were no object-type features, so there was no need to label-encode any columns.

c. Feature engineering
- Adding basic date features: year, month, week number, day of the year, day of the week, hour.
- Adding cyclical features: for years, months, day of the week, and hour.
- Adding seasons based on the Month of the Year.
- Indicating weekends: added a feature to indicate whether the day is a weekend.
- Adding day/night indicator: a feature was added to indicate whether the time is during the day or night, which varies depending on the month.
- Incorporating public holidays: added a feature to include public holidays in the Estonian calendar.


### 2.2 - Splitting the Data into a Train Set and a Test Set

Given the large size of the dataset (approximately 2 million rows), I decided to use the following proportions:
- Train Set: 70% of the total dataset length
- Test Set: 30% of the total dataset length

Since I’m working with time series data, I didn’t shuffle the data. This approach allows the models to better learn the trends and variations that may occur over time. Of course, each of these sets was then split into an X set (features) and a y set (target).


### 2.3 - Data scaling

Scaling of my training data was only necessary for the Linear Regression model, as it is sensitive to distances and thus to the scale of the data.



### 2.4 - Models used

I decided to try and compare seven different models in this project. Initially, I did not focus on optimizing their parameters. My strategy was to first select the best-performing models in their raw form and then proceed with optimization.

1. **"Dummy" model:**: this very simple model serves as a baseline for comparing the effectiveness of my other, more advanced models. It represents the mean of the target values from my training data. In other words: Baseline = mean(y_train)

2. **"Engineer" model:**: this model is a more advanced version of the baseline. It’s a model that could be quickly conceived by engineers or students. It’s built as follows:

- Electricity consumption prediction: to predict electricity consumption, I assumed that the values from the previous year would be used for the prediction dates. For example, the electricity consumption on April 5, 2023 (a value to be predicted) is assumed to be the same as that on April 5, 2022. Although electricity consumption depends on many factors, including weather, I assume that year-to-year variations are negligible for this model.

- Electricity production prediction: to predict electricity production, I examined a correlation matrix to identify which feature was strongly correlated with electricity production. This correlation showed that electricity production was strongly related to the "direct_solar_radiation" feature, which makes sense since production in the data is exclusively via solar panels. Therefore, I modeled the predicted consumption values using a simple ratio with the "direct_solar_radiation" feature.

3. **Linear regression:** I started with linear regression because it's simple and interpretable. It helped me understand the basic relationship between variables. Although it’s not the most powerful model, it’s a great baseline that provides insights into which features might be influential.

4. **Random Forest:** I used a random forest model to capture more complex interactions between features. Indeed, Random forest models are great for handling a mix of numerical and categorical data and they can model non-linear relationships. It's a very popular model to try in a data science projetc. It’s particularly useful when there’s a lot of noise in the data because it’s less likely to overfit compared to other models.

5. **XGBoost:** To push the accuracy further, I decided to use a XGBoost model. This model was very popular in other competitor projects. It’s a powerful boosting algorithm that usually outperforms random forests by iteratively improving the predictions. In this project, XGBoost was useful because it handled the large amount of data well and was effective at capturing intricate patterns in the electricity consumption data.

6. **LGBM:** Finally, I added LGBM (LightGBM) to the mix. It’s similar to XGBoost but faster and more efficient, especially with large datasets. LGBM was particularly helpful in speeding up the training process without sacrificing performance, which was important given the size of the data I was working with. This model was slightly better than the XGBoost model.

7. **Optimized LGBM:**: since the LGBM model was the most effective, I decided to optimize it using grid search. The process was guided by minimizing the MAE (Mean Absolute Error) on the validation set.



## 3 - Results

Below are the performance results of the different models on the test set. For reference, the train set represents 70% of the data, while the test set constitutes 30%, and the data was not shuffled. The metric used to compare the models is Mean Absolute Error (MAE).


| Models                  | MAE Scores                          |
|-------------------------|-------------------------------------|
| Dummy model             | 444.129                             |
| Engineer model          | Only used for graphic visualization |
| Linear regression       | 414.55                              |
| Random Forest           | 101.92                              |
| XGboost                 | 109.25                              |
| LGBM                    | 92.93                               |
| LGBM optimized 1        | 88.34                               |
| LGBM optimized 2        | 44.72                               |
|------------------------ |-------------------------------------|
| Competition winner      | 52.3                                |


Several points are worth noting from the results:

1. **Advanced models vs. baseline:** it is evident that the more advanced models perform significantly better compared to the baseline.

2. **Linear regression performance:** the Linear Regression model did not perform well compared to the baseline. It is likely that the data scaling was not precise enough, which may have negatively impacted the model's performance.

3. **LGBM model optimization:** The LGBM model, which performed the best, was optimized twice. First, using grid search, which slightly improved the score. The second improvement came when the code was refined in section 3, specifically by adding the function y_pred = y_pred.clip(0) to set all negative predictions to zero. Since negative consumption or production values are not feasible, this adjustment greatly improved the model's accuracy.

4. **Comparison with winner’s model:** While the winner's model might appear less performant than my LGBM optimized 2 model, the winner's model was evaluated on recent 2024 data connected directly to the company's API. Due to the competition being closed, I was unable to replicate this. My evaluation was limited to the Test set (30% of the final dataset), which may explain some overfitting on the training data.



## 4 - Areas of improvement

De nombreux axes d'amelioration aurai

1. **Creating separate models:**: I think it might have been beneficial to develop two distinct models for this project, one focusing on predicting consumption and the other on production. Since production and consumption are independent, exploring this approach could have potentially improved the model’s performance.

2. **Ensemble methods:**: many top competitors on Kaggle have used sequential or ensemble methods to enhance prediction accuracy. Although this approach is technical and time-consuming, it could have been explored in my project to achieve better results.

3. **API testing:**: I would have liked to test my model via the company’s API to get an exact score for the project and obtain an official ranking on Kaggle. Unfortunately, this was not possible when I started the competition as it was closed.

4. **Code improvement:**: the final code could have been improved. Many highly-ranked competitors used classes in their projects, which can elevate code quality and make it more professional. I am currently not proficient with classes, but learning to use them could enhance the project.



## 5 - Contribution

It is challenging to list all the sources that contributed to this project. I learned a lot from the various notebooks shared by competition participants. Numerous online searches and YouTube tutorials also helped me overcome technical challenges I encountered. Additionally, AI tools played a significant role in enhancing my code and research, allowing me to achieve a higher level of proficiency and to improve my skills in handling and analyzing time series data, which was my goal at the start of this competition.
