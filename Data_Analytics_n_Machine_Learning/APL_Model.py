from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import log1p, col

# Init Spark
spark = SparkSession.builder.appName("HDB_Resale_Price_Prediction_GBT").getOrCreate()

# Load parquet file on GCS
gcs_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df = spark.read.parquet(gcs_path)

# ---------------------------------------
# Feature Engineering
# ---------------------------------------

# Add new features
df = df.withColumn("floor_area_log", log1p("floor_area_sqm"))
df = df.withColumn("price_per_sqm", col("resale_price") / col("floor_area_sqm"))

# StringIndexer
categorical_cols = ["town", "flat_type", "flat_model"]
indexers = [
    StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
    for col_name in categorical_cols
]

numeric_cols = ["floor_area_sqm", "floor_area_log", "flat_age", "storey_median", "month_num", "year"]

# Combine all features
feature_cols = [f"{col}_index" for col in categorical_cols] + numeric_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ---------------------------------------
# Build GPT Model
# ---------------------------------------

# GBT (Gradient Boosted Trees)
gbt = GBTRegressor(featuresCol="features", labelCol="resale_price", maxDepth=10, maxIter=100, stepSize=0.1, maxBins=64)

# Build pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt])

# Split datasets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train data nbr: {train_data.count()}")
print(f"Test data nbr: {test_data.count()}")

# Train and save the model
model = pipeline.fit(train_data)
model.save("gs://hrpaps-bucket/models/hrpaps_gbt_pipeline_model")

# Run simple predictions
predictions = model.transform(test_data)

print("Testing sample on test dataset")
predictions.select("resale_price", "prediction").show(10)

# ---------------------------------------
# Run Evaluations
# ---------------------------------------

evaluator_rmse = RegressionEvaluator(
    labelCol="resale_price", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(
    labelCol="resale_price", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"\n Model evaluation result:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# ---------------------------------------
# Feature Importance
# ---------------------------------------

gbt_model = model.stages[-1]
importances = gbt_model.featureImportances
feature_importance_list = list(zip(assembler.getInputCols(), importances.toArray()))
feature_importance_list = sorted(feature_importance_list, key=lambda x: -x[1])

print("\n Ranking of Feature Importance: ")
for feature, score in feature_importance_list:
    print(f"{feature}: {score:.4f}")

# End spark session
spark.stop()
