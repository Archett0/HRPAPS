from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit, rand, log1p, col
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import PipelineModel
import random
import datetime

def generate_realistic_test_data(spark: SparkSession, source_df: DataFrame, months=24, samples_per_month=100) -> DataFrame:
    """
    :param spark: SparkSession
    :param source_df: Org HDB resale Data
    :param months: Months to generate
    :param samples_per_month
    :return: PySpark DataFrame
    """

    towns = [row['town'] for row in source_df.select("town").distinct().collect()]

    flat_type_dist = source_df.groupBy("flat_type").count().collect()
    flat_type_choices = [row["flat_type"] for row in flat_type_dist]
    flat_type_weights = [row["count"] for row in flat_type_dist]

    flat_model_dist = source_df.groupBy("flat_model").count().collect()
    flat_model_choices = [row["flat_model"] for row in flat_model_dist]
    flat_model_weights = [row["count"] for row in flat_model_dist]

    area_stats = (
        source_df.groupBy("flat_type")
        .agg({"floor_area_sqm": "avg"})
        .withColumnRenamed("avg(floor_area_sqm)", "mean_area")
        .collect()
    )
    flat_type_area_map = {row["flat_type"]: row["mean_area"] for row in area_stats}

    age_stats = (
        source_df.groupBy("town")
        .agg({"flat_age": "avg"})
        .withColumnRenamed("avg(flat_age)", "avg_flat_age")
        .collect()
    )
    town_avg_age_map = {row["town"]: int(row["avg_flat_age"]) for row in age_stats}

    storeys = [int(row["storey_median"]) for row in source_df.select("storey_median").dropna().distinct().collect()]
    min_storey, max_storey = min(storeys), max(storeys)

    data = []
    start_date = datetime.date.today().replace(day=1)

    for month_offset in range(months):
        month_date = (start_date + datetime.timedelta(days=month_offset * 30))
        year = month_date.year
        month_num = month_date.month
        month_str = f"{year}-{month_num:02d}"

        for _ in range(samples_per_month):
            town = random.choice(towns)
            flat_type = random.choices(flat_type_choices, weights=flat_type_weights, k=1)[0]
            flat_model = random.choices(flat_model_choices, weights=flat_model_weights, k=1)[0]

            # floor area based on flat_type
            base_area = flat_type_area_map.get(flat_type, 90)
            floor_area_sqm = round(random.gauss(base_area, 5), 1)
            floor_area_sqm = max(35.0, min(160.0, floor_area_sqm))

            # storey
            storey_median = random.randint(min_storey, max_storey)

            # flat_age based on town
            flat_age = max(1, min(60, int(random.gauss(town_avg_age_map.get(town, 30), 5))))

            data.append((month_str, year, month_num, town, flat_type, flat_model,
                         floor_area_sqm, storey_median, flat_age))

    columns = ["month", "year", "month_num", "town", "flat_type", "flat_model",
               "floor_area_sqm", "storey_median", "flat_age"]

    return spark.createDataFrame(data, schema=columns)

# ------------------------------------------------------------
# Generate data for prediction
# ------------------------------------------------------------
spark = SparkSession.builder.appName("Generate_HDB_Test_Data").getOrCreate()

gcs_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df = spark.read.parquet(gcs_path)

test_df = generate_realistic_test_data(spark, df, months=24, samples_per_month=100)

test_data_output_path = "gs://hrpaps-bucket/predictions/data/generated_test_data.parquet"
test_df.write.mode("overwrite").parquet(test_data_output_path)

# ------------------------------------------------------------
# Run predictions
# ------------------------------------------------------------
MODEL_PATH = "gs://hrpaps-bucket/models/hrpaps_gbt_pipeline_model"
TEST_DATA_PATH = "gs://hrpaps-bucket/predictions/data/generated_test_data.parquet"
PREDICTION_OUTPUT_PATH = "gs://hrpaps-bucket/predictions/predictions_output.parquet"
PREDICTION_OUTPUT_CSV_PATH = "gs://hrpaps-bucket/predictions/predictions_csv"
PREDICTION_OUTPUT_JSON_PATH = "gs://hrpaps-bucket/predictions/predictions_json"

model = PipelineModel.load(MODEL_PATH)

test_df = spark.read.parquet(TEST_DATA_PATH)
test_df = test_df.withColumn("floor_area_log", log1p(col("floor_area_sqm")))

predictions = model.transform(test_df)

# ------------------------------------------------------------
# Save prediction results
# ------------------------------------------------------------

predictions.select("month", "town", "flat_type", "flat_model",
                   "floor_area_sqm", "storey_median", "flat_age",
                   "prediction") \
           .write.mode("overwrite") \
           .parquet(PREDICTION_OUTPUT_PATH)

predictions.select("month", "town", "flat_type", "flat_model",
                   "floor_area_sqm", "storey_median", "flat_age",
                   "prediction") \
           .coalesce(1) \
           .write.mode("overwrite") \
           .option("header", True) \
           .csv(PREDICTION_OUTPUT_CSV_PATH)

predictions.select("month", "town", "flat_type", "flat_model",
                   "floor_area_sqm", "storey_median", "flat_age",
                   "prediction") \
           .coalesce(1) \
           .write.mode("overwrite") \
           .json(PREDICTION_OUTPUT_JSON_PATH)

spark.stop()
