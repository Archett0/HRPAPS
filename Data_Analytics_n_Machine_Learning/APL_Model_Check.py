from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import log1p, col
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# Init spark
spark = SparkSession.builder.appName("HDB_Resale_Price_Prediction_GBT_TEST1").getOrCreate()

# Load parquet file on GCS
gcs_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df = spark.read.parquet(gcs_path)

# ---------------------------------------
# Feature Engineering
# ---------------------------------------

# Add new features
df = df.withColumn("floor_area_log", log1p("floor_area_sqm"))
df = df.withColumn("price_per_sqm", col("resale_price") / col("floor_area_sqm"))

# Split datasets
_, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"Test data nbr: {test_data.count()}")

# ---------------------------------------
# Load Model from GCS
# ---------------------------------------
model = PipelineModel.load("gs://hrpaps-bucket/models/hrpaps_gbt_pipeline_model")

# Run predictions
predictions = model.transform(test_data)

# ---------------------------------------
# Save Predictions to GCS
# ---------------------------------------
predictions_output_path = "gs://hrpaps-bucket/logs/predictions_output.csv"
predictions.select("resale_price", "prediction").write.csv(predictions_output_path, header=True, mode="overwrite")

# ---------------------------------------
# Run Evaluations
# ---------------------------------------

evaluator_rmse = RegressionEvaluator(
    labelCol="resale_price", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(
    labelCol="resale_price", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

# ---------------------------------------
# Save Evaluation Results to GCS
# ---------------------------------------
evaluation_results = Row(rmse=rmse, r2=r2)
evaluation_results_df = spark.createDataFrame([evaluation_results])

evaluation_results_output_path = "gs://hrpaps-bucket/logs/evaluation_results.csv"
evaluation_results_df.coalesce(1).write.csv(evaluation_results_output_path, header=True, mode="overwrite")

# End spark session
spark.stop()
