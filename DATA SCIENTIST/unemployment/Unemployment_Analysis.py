# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:10:41 2026

@author: hp
"""

# TASK 2: Unemployment Analysis with Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = ("unemploymentData.csv")

# Preview dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:\n", data.head())

# 2. Data Cleaning
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Ensure unemployment rate is numeric
data['UnemploymentRate'] = pd.to_numeric(data['UnemploymentRate'], errors='coerce')

# Drop missing values
data = data.dropna()

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='UnemploymentRate', data=data, color='blue')
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Distribution plot
plt.figure(figsize=(8,5))
sns.histplot(data['UnemploymentRate'], bins=20, kde=True, color='orange')
plt.title("Distribution of Unemployment Rates")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.show()

# 4. Impact of Covid-19
covid_period = data[(data['Date'] >= '2020-03-01') & (data['Date'] <= '2021-12-31')]
pre_covid = data[data['Date'] < '2020-03-01']

print("\nAverage Unemployment Pre-Covid:", round(pre_covid['UnemploymentRate'].mean(), 2))
print("Average Unemployment During Covid:", round(covid_period['UnemploymentRate'].mean(), 2))

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='UnemploymentRate', data=pre_covid, label="Pre-Covid", color='green')
sns.lineplot(x='Date', y='UnemploymentRate', data=covid_period, label="Covid Period", color='red')
plt.title("Unemployment Trends: Pre-Covid vs Covid Period")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# 5. Seasonal Trends
data['Month'] = data['Date'].dt.month
monthly_avg = data.groupby('Month')['UnemploymentRate'].mean()

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="viridis")
plt.title("Average Unemployment Rate by Month")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.show()

# 6. Insights
print("\n--- Insights ---")
print("1. Covid-19 caused a sharp rise in unemployment rates compared to pre-Covid averages.")
print("2. Certain months show consistently higher unemployment, indicating possible seasonal effects.")
print("3. These insights can inform economic and social policies, e.g., targeted job support programs.")
