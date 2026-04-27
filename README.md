# Income Classification using PySpark ML Pipelines

## 📌 Overview
This project builds a scalable machine learning pipeline using PySpark to classify income levels based on demographic and employment data. The solution demonstrates end-to-end data processing, feature engineering, and model training using distributed computing.

## 🚀 Objectives
- Perform data cleaning and preprocessing on structured data
- Build scalable ML pipelines using PySpark
- Train and compare multiple classification models
- Evaluate model performance using multiple metrics

## 🛠️ Tools & Technologies
- Python
- PySpark (Spark ML)
- Machine Learning (Decision Tree, Random Forest)
- Data Processing & Feature Engineering

## 📊 Project Workflow
1. Data Ingestion (CSV loading with Spark)
2. Data Cleaning (handling missing values, column formatting)
3. Feature Engineering:
   - StringIndexer for categorical variables
   - OneHotEncoder for encoding
4. Feature Assembly using VectorAssembler
5. Model Building:
   - Decision Tree Classifier
   - Random Forest Classifier
6. Model Evaluation:
   - Accuracy
   - F1 Score
   - Precision
   - Recall

## 📈 Results
- Random Forest outperformed Decision Tree in overall classification accuracy and generalisation.
- Feature engineering significantly improved model performance.
- Spark pipelines enabled efficient and reproducible workflows.

## 📂 Project Structure
