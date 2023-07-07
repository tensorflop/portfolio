<img src="challenge.png">

This project was a competition put on by Coursera. It was a quick competition with only three days open for submissions. I achieved a result in the top 1% of all competitors.
## Challenge Description
The goal of this challenge was to predict the probability of customer churn for a streaming service. The performance metric for this challenge was the Area Under the Curve (AUC) of the Receiver-Operator Characteristic (ROC). A train and test dataset were provided, with the training dataset containing labels for customer churn. The data description is shown in the table below.
<p align="center">
<img src="datadescription.png">
</p>

## General Approach
My general approach to data science competition projects is:
1. Exploratory Data Analysis (EDA)
2. Data Preparation
3. Baseline Model Performance
4. Cross Validation Strategy
5. Model Selection
6. Feature Engineering
7. Model Tuning
That flow is pretty well suited to competitions. Real data science can have a much more complicated 

## Exploratory Data Analysis
I like to use ydata profiling, a Python package that can generate a pretty nice html report with one line of code. The report gives a nice summary of each feature that includes some great information such as:
* Number of missing values
* Number of unique values
* Number of zeroes
* Max, min, and mean
* Histogram
* Extreme values
The dataset for this competition was pretty straightforward. There were no missing values for 

## Data Preparation
