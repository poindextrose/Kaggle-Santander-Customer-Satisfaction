# Introduction

This repo contains scripts for competing in the [Kaggle Santanter Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) competition.

# Starting

The Jupyter Notebook [data cleaning.ipynb](https://github.com/poindextrose/Kaggle-Santander-Customer-Satisfaction/blob/master/data%20cleaning.ipynb) is the starting point to clean up the data. Run this first to prep the data for the other scripts.

# Models

I tried out the following algorithms in the various scripts:
* XGBoost
* TPOT
* GradientBoostingClassifier
* SVM

I ran these models on a few machines and had the results saved to a cloud MongoDB so I could find the best hyperparameters.
