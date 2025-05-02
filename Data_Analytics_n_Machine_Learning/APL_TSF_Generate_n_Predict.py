from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import datetime

spark = SparkSession.builder.appName("HDB_Resale_TSF_Predict_Future").getOrCreate()

gcs_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df_spark = spark.read.parquet(gcs_path)
df = df_spark.toPandas()

df['month'] = pd.to_datetime(df['month'])
df['year_month'] = df['month'].dt.to_period('M')
df['psm'] = df['resale_price'] / df['floor_area_sqm']

le = LabelEncoder()
df['town_index'] = le.fit_transform(df['town'])

agg_df = df.groupby(['year_month', 'town']).agg({
    'resale_price': 'mean',
    'floor_area_sqm': 'mean',
    'flat_age': 'mean',
    'storey_median': 'mean',
    'psm': 'mean',
    'town_index': 'first'
}).reset_index()

agg_df['year_month'] = agg_df['year_month'].dt.to_timestamp()
agg_df = agg_df.sort_values(['town', 'year_month'])

forecast_months = 24
future_results = []

towns = agg_df['town'].unique()

model = xgb.XGBRegressor()
model.load_model("gs://hrpaps-bucket/models/xgboost_resale_model_mixed.json")

for town in towns:
    town_df = agg_df[agg_df['town'] == town].copy().sort_values('year_month')

    for _ in range(forecast_months):
        recent = town_df.tail(3)
        if len(recent) < 3:
            print(f"Jumping {town} bc insufficient data.")
            break

        lag_1 = recent.iloc[-1]['resale_price']
        lag_2 = recent.iloc[-2]['resale_price']
        lag_3 = recent.iloc[-3]['resale_price']

        last_row = recent.iloc[-1]

        last_month = last_row['year_month']
        new_month = (last_month + pd.DateOffset(months=1)).replace(day=1)

        new_entry = {
            'year_month': new_month,
            'town': town,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'floor_area_sqm': last_row['floor_area_sqm'],
            'flat_age': last_row['flat_age'] + 1/12,
            'storey_median': last_row['storey_median'],
            'psm': last_row['psm'],
            'month_num': new_month.month,
            'year': new_month.year,
            'town_index': last_row['town_index']
        }

        X_pred = pd.DataFrame([new_entry])[[
            'lag_1', 'lag_2', 'lag_3',
            'floor_area_sqm', 'flat_age', 'storey_median',
            'psm', 'month_num', 'year', 'town_index'
        ]]
        y_pred = model.predict(X_pred)[0]

        new_entry['resale_price'] = y_pred
        future_results.append(new_entry)

        town_df = pd.concat([town_df, pd.DataFrame([new_entry])], ignore_index=True)

future_df = pd.DataFrame(future_results)
future_df = future_df[['year_month', 'town', 'resale_price']].sort_values(['town', 'year_month'])

spark_future_df = spark.createDataFrame(future_df)
spark_future_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://hrpaps-bucket/predictions/hdb_forecast_next_24_months.csv")
print("Prediction saved as CSV")
