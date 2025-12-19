# MOOC Student Certification Prediction using Machine Learning

## Project Overview

This repository contains the final project for **Machine Learning**, focused on predicting whether a student will successfully complete (certified) a Massive Open Online Course (MOOC). The project addresses a real‑world educational analytics problem where **class imbalance** and **heterogeneous features** pose significant modeling challenges.

Multiple **classical machine learning models** and a **deep learning model (FCNN)** were implemented, tuned, and fairly compared using a consistent preprocessing and evaluation pipeline. The primary evaluation metric is **F1‑score**, as accuracy alone is misleading for highly imbalanced datasets.


## Dataset

* **Source_of_dataset**: Kaggle – MOOC Student Online Dataset (2023–24)
* **Original Size**: 416,921 records, 22 features
* **Sample Used**: 80,000 records (randomly selected for computational feasibility)
* **Target Variable**: `certified`

  * `1` → Student completed the course
  * `0` → Student did not complete the course

### Feature Types

* **Categorical**: institute, course_id, semester, country, education level, gender
* **Numerical / Behavioral**: events count, active days, video plays, forum posts, chapters accessed
* **Time‑based**: start time and last activity time

The dataset is **highly imbalanced**, with significantly very fewer certified students.

## Data Preprocessing & Feature Engineering

The following steps were applied consistently across all the models:

* Dropped the leakage prone columns: `Id`, `userid_DI`, `grade`
* One‑hot encoding for categorical variables
* Country feature grouped into **USA , India and Other** to reduce high cardinality
* Datetime cleaning and feature engineering: Created `days_active_total` from start and last activity timestamps
* Log transformation applied to skewed behavioral features
* Age values clipped to remove unrealistic outliers
* Final feature count expanded from **22 → 44 features**

### Scaling & Imbalance Handling

* **StandardScaler** applied to numerical features
* Scaling fitted on training data only (to avoid data leakage)
* **SMOTE** used to handle severe class imbalance


## Models Implemented

Each notebook represents a **separate experiment**, using the same preprocessing pipeline to ensure fair comparison.
_________________________________________________________________________________________________
| Model                                 | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| Logistic Regression                   | Baseline linear classifier with hyperparameter tuning |
| Support Vector Machine (RBF)          | Non‑linear margin‑based classifier                    |
| K‑Nearest Neighbors                   | Distance‑based instance learner                       |
| Decision Tree                         | Rule‑based tree model with pruning                    |
| Random Forest                         | Ensemble of decision trees (**best model**)           |
| Fully Connected Neural Network (FCNN) | Deep learning model with regularization               |
_________________________________________________________________________________________________

## Model Evaluation (Test Set)

Since the dataset is imbalanced, **F1‑score** is treated as the primary metric.
___________________________________________________________________________
| Model               | Accuracy   | Precision  | Recall     | F1‑Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9682     | 0.5307     | 0.9651     | 0.6848     |
| SVM (RBF)           | 0.9735     | 0.5881     | 0.8698     | 0.7016     |
| KNN                 | 0.9791     | 0.6642     | 0.8419     | 0.7426     |
| Decision Tree       | 0.9792     | 0.6530     | 0.8930     | 0.7544     |
| **Random Forest**   | **0.9847** | **0.7181** | **0.9419** | **0.8149** |
| FCNN                | 0.9812     | 0.6688     | 0.9448     | 0.7830     |
___________________________________________________________________________
**Random Forest achieved the best overall performance**, balancing precision and recall effectively.

## FCNN Experiments

The FCNN required additional care due to instability caused by imbalanced data:

* Rescaling applied **after SMOTE** to stabilize gradients
* Baseline network: 2 hidden layers (64, 32) with ReLU + Dropout
* Three experimental variants:

  1. Smaller network
  2. L2 regularization
  3. Batch Normalization (**best FCNN variant**)

The final FCNN performed competitively but did not surpass Random Forest.

## Key Findings

* Accuracy alone is misleading for imbalanced datasets
* SMOTE, scaling, and feature engineering were critical
* Tree‑based and ensemble models captured non‑linear patterns effectively
* Random Forest generalized best with minimal overfitting
* FCNN required normalization and regularization to remain stable

## Tech Stack

* Python
* Pandas, NumPy
* Scikit‑learn
* TensorFlow / Keras
* Imbalanced‑learn (SMOTE)
* Matplotlib / Seaborn

## How to Run

Open the notebooks in **Google Colab** or **Jupyter Notebook** and can directly run the notebooks.

## References

* Kaggle MOOC Dataset: [https://www.kaggle.com/datasets/kanikanarang94/mooc-dataset](https://www.kaggle.com/datasets/kanikanarang94/mooc-dataset)
* Scikit‑learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
* TensorFlow / Keras: [https://www.tensorflow.org](https://www.tensorflow.org)

------

## Author

**Sai Tejaswi Kondapally**
Machine Learning 
Computer Science Student
