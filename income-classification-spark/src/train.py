from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from preprocess import build_preprocessing_pipeline
from pyspark.ml.feature import StringIndexer


def main():
    spark = SparkSession.builder.appName("IncomeClassification").getOrCreate()

    # Load data
    df = spark.read.csv("data/income.csv", header=True, inferSchema=True)

    # Clean column names
    df = df.select([col(c).alias(c.strip()) for c in df.columns])

    label_col = "income_class"
    df = df.dropna(subset=[label_col])

    # Preprocessing
    indexers, encoders, assembler = build_preprocessing_pipeline(df, label_col)
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid='keep')

    # Models
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    # Pipelines
    dt_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, dt])
    rf_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])

    # Train-test split
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    # Train
    dt_model = dt_pipeline.fit(train_data)
    rf_model = rf_pipeline.fit(train_data)

    # Predict
    dt_preds = dt_model.transform(test_data)
    rf_preds = rf_model.transform(test_data)

    # Evaluation
    metrics = ["accuracy", "f1", "weightedPrecision", "weightedRecall"]

    for metric in metrics:
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
        print(f"Decision Tree {metric}: {evaluator.evaluate(dt_preds):.4f}")
        print(f"Random Forest {metric}: {evaluator.evaluate(rf_preds):.4f}")

    # Feature importance
    print("Random Forest Feature Importances:")
    print(rf_model.stages[-1].featureImportances)

    # Save model
    rf_model.write().overwrite().save("models/random_forest_model")

    spark.stop()


if __name__ == "__main__":
    main()