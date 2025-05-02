from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, regexp_extract, when
import pandas as pd

spark = SparkSession.builder.appName("FullPipeline").getOrCreate()

pdf = pd.read_parquet("gs://hrpaps-bucket/test/cleaned_hdb_resale_data.parquet")
df = spark.createDataFrame(pdf)

for col_name, dtype in df.dtypes:
    if dtype in ["timestamp", "date", "string", "binary"]:
        continue
    elif dtype == "boolean":
        df = df.withColumn(col_name, col(col_name).cast("string"))
    elif dtype.startswith("int") or dtype.startswith("double") or dtype.startswith("float"):
        continue
    else:
        df = df.withColumn(col_name, col(col_name).cast("string"))

df = df.withColumn("year", substring(col("month"), 1, 4).cast("int"))
df = df.withColumn("month_num", substring(col("month"), 6, 2).cast("int"))
df = df.withColumn("flat_age", (2025 - col("lease_commence_date")).cast("int"))

df = df.withColumn("storey_low", regexp_extract(col("storey_range"), r"(\d+)\D+(\d+)", 1).cast("int"))
df = df.withColumn("storey_high", regexp_extract(col("storey_range"), r"(\d+)\D+(\d+)", 2).cast("int"))

df = df.withColumn(
    "storey_median",
    when(
        col("storey_high").isNotNull() & (col("storey_high") != 0),
        (col("storey_low") + col("storey_high")) / 2
    ).otherwise(col("storey_low"))
)

df = df.withColumn(
    "area_category",
    when(col("floor_area_sqm") < 70, "small")
    .when(col("floor_area_sqm") < 100, "medium")
    .otherwise("large")
)

df = df.withColumn("price_per_sqm", (col("resale_price") / col("floor_area_sqm")).cast("double"))

output_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df.coalesce(1).write.mode("overwrite").parquet(output_path)

spark.stop()
