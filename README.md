# HealthDataMLModel
This project looks at my personal health data collected by Apple. The goal of this project was to see if I could estimate whether I'd have a good nights sleep based on the workouts I did that day. The code in this repo supports two blog posts that were written to walk someone new to machine learning through the process by using a fun comparison to an RPG video game. \
[A NOBLE QUEST: CREATING A PREDICTIVE MODEL WITH HEALTH DATA - PART 1](https://insights.jahnelgroup.com/a-noble-quest-creating-a-predictive-model-with-health-data) \
[A NOBLE QUEST: CREATING A PREDICTIVE MODEL WITH HEALTH DATA- PART 2](https://insights.jahnelgroup.com/a-noble-quest-creating-a-predictive-model-with-health-data-part-2) 

# Models
To accomplish this 3 classification models wwere created and compared; XGBoost, Randomdom Forest, and Logistic.

# Tuning
XGBoost utilized gridsearch to find the optimal metrics. The Logistic Model was tuned by examining the impacts of oversampling using SMOTE and adjusting the prediction threshold.

# DataSet
The data was requested from Apple in the Health App and was sent as an XML file. From that file workout and sleep data were mined.


