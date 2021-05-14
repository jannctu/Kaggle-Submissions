# Titanic - Machine Learning from Disaster
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.6.1/css/all.css" integrity="sha384-gfdkjb5BdAXd+lj+gudLWI+BXq4IuLW5IT+brZEZsLFm++aCMlF1V92rMkPaX4PP" crossorigin="anonymous">

## 🎯 Introduction
Titanic is legendary ML competition in [Kaggle](https://www.kaggle.com/c/titanic/overview). 

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

## 💘 Goal 
It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.
## 💋 Approach
I used classical Logistic Regression for classfication and GridSearch to choose the hyperparameters.

## 👀 Evaluation 

The score is the percentage of passengers you correctly predict. This is known as [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification).

    Accuracy = (TP + TN)/(TP + TN + FP + FN)

where: `TP = True positive; FP = False positive; TN = True negative; FN = False negative`


