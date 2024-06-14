# Heart Disease Classification
![](https://st1.thehealthsite.com/wp-content/uploads/2021/09/Heart.jpeg?impolicy=Medium_Widthonly&w=400)
Welcome to the **Heart Disease Classification** project! This repository contains a data science project aimed at predicting the presence of heart disease based on various health-related features using machine learning techniques. We have followed rigorous data preprocessing, model training, and evaluation steps to develop accurate predictive models.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Motivation](#motivation)
4. [Observations from Descriptive Statistics](#observations-from-descriptive-statistics)
5. [Techniques and Models Used](#techniques-and-models-used)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Automation with Shell Script](#automation-with-shell-script)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The goal of the Heart Disease Classification project is to predict the presence or absence of heart disease in individuals based on their health attributes. We employ various machine learning algorithms and techniques to achieve this objective, ensuring robust and accurate predictions.

## Data Description

The dataset used in this project includes records of individuals with the following features:

1. **age**: Age of the individual.
2. **gender**: Gender of the individual (male or female).
3. **cp (chest pain type)**: Type of chest pain experienced (categorical variable).
4. **trestbps (resting blood pressure)**: Resting blood pressure in mm Hg.
5. **chol (cholesterol)**: Serum cholesterol in mg/dL.
6. **fps (fasting blood sugar)**: Whether fasting blood sugar > 120 mg/dL (binary variable).
7. **restecg (resting electrocardiographic results)**: Resting electrocardiographic results (categorical variable).
8. **thalach (maximum heart rate achieved)**: Maximum heart rate achieved.
9. **exang (exercise-induced angina)**: Exercise-induced angina (binary variable).
10. **oldpeak**: ST depression induced by exercise relative to rest.
11. **slope**: The slope of the peak exercise ST segment (categorical variable).
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy.
13. **thal**: Thalassemia (categorical variable: 3 = normal; 6 = fixed defect; 7 = reversible defect).
14. **class**: Target variable indicating the presence of heart disease (5 classes).

## Motivation

The primary motivation behind building this model is to predict the likelihood of heart disease based on individual health characteristics. Key reasons for developing such a model include:

1. **Early Detection and Intervention**: Identifying individuals at high risk of heart disease early allows for timely medical interventions and personalized treatment plans, potentially improving patient outcomes and reducing healthcare costs.

2. **Personalized Healthcare**: Tailoring healthcare strategies based on predicted risk levels helps in providing personalized recommendations and preventive measures to individuals, promoting healthier lifestyles and reducing disease burden.

3. **Public Health Impact**: Insights gained from the model can inform public health policies and initiatives aimed at reducing the prevalence of heart disease through targeted awareness campaigns and screening programs.

4. **Research and Insights**: Analyzing the relationships between health attributes and heart disease outcomes provides valuable insights for medical research, potentially uncovering new factors influencing cardiovascular health.

## Observations from Descriptive Statistics

1. **Age**: The average age is 54.44 years, with a range from 29 to 77 years, indicating a middle-aged to older adult population.
2. **Resting Blood Pressure (trestbps)**: The mean resting blood pressure is 131.69 mm Hg, with values ranging from 94 to 200 mm Hg, showing a mix of normal and hypertensive individuals.
3. **Cholesterol (chol)**: The average cholesterol level is 246.69 mg/dL, with a wide range from 126 to 564 mg/dL, indicating high variability and some extremely high cholesterol levels.
4. **Maximum Heart Rate Achieved (thalach)**: The mean maximum heart rate is 149.61 bpm, ranging from 71 to 202 bpm, reflecting differences in cardiovascular fitness levels.
5. **ST Depression Induced by Exercise (oldpeak)**: The average ST depression is 1.04, with values from 0 to 6.2, showing that while many individuals have no depression, a significant number have higher values indicating varying degrees of exercise-induced ischemia.

## Techniques and Models Used

1. **Data Cleaning**: Removal of outliers, handling missing values, and ensuring data consistency.
2. **Feature Engineering**: Creating new features or transforming existing ones to improve model performance.
3. **Machine Learning Models**:
   - Logistic Regression
   - Decision Trees
   - Naive Bayes
   - Support Vector Machines (SVM)
   - Random Forests

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```sh
git clone https://github.com/yourusername/Heart_Disease_Classification.git
cd Heart_Disease_Classification
conda create --name heart_disease_env python=3.8
conda activate heart_disease_env
pip install -r requirements.txt
```

## Usage

Follow these steps to run the project:

1. **Preprocess the Data**: Explore the data using a Jupyter notebook, clean it, and save the cleaned data.
2. **Save File Paths**: Update the paths of the cleaned files in the `config.py` file.
3. **Create Folds**: Run the `create_fold.py` file to create data folds.
    ```sh
    python create_fold.py
    ```
4. **Train the Model**: Execute the `baseline_model.py` script with the desired arguments.
    ```sh
    python baseline_model.py --fold 5 --model lr --logs logisticRegression_model --scale True --metric roc_auc --plot_roc True
    ```

### Arguments for `baseline_model.py`

- `--fold`: Number of folds for cross-validation (default: 5)
- `--model`: Model type (`lr`, `dt`, `nb`)
- `--logs`: Path to save logs (default: None)
- `--scale`: Whether to scale the data (default: False)
- `--metric`: Metric for evaluation (`f1_score`, `roc_auc`, `precision`, `recall`, `accuracy`, `f1_weighted`)
- `--plot_roc`: Whether to plot ROC curve (default: False)

## Automation with Shell Script

To automate the training process, you can use the provided shell script `train_run.sh`. This script will run the training commands for different models automatically.

Create the `train_run.sh` file with the following content:

```sh
#!/bin/sh
set -e
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log "TRAINING STARTED!"
log "Author: Your Name"

echo "LOGISTIC REGRESSION TRAINING STARTED!"
python model.py --fold 10 --model lr --logs logisticRegression_model --scale True --metric roc_auc --plot_roc True
echo "SUCCESSFUL DONE!"

echo "================================================================================================================"

echo "DECISION TREE TRAINING STARTED!"
python model.py --fold 10 --model dt --logs decisionTree_model --scale True --metric roc_auc --plot_roc True
echo "SUCCESSFUL DONE!"

echo "================================================================================================================"

echo "NAIVE BAYES TRAINING STARTED!"
python model.py --fold 10 --model nb --logs naiveBayes_model --scale True --metric roc_auc --plot_roc True
echo "SUCCESSFUL DONE!"
```

Make the script executable and run it:

```sh
chmod +x train_run.sh
./train_run.sh
```

## Contributing

Contributions to improve this project are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

