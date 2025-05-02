from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

spark = SparkSession.builder.appName("HDB_Resale_TSF_Training_MixedFeatures").getOrCreate()

gcs_path = "gs://hrpaps-bucket/test/forMachineLearning.parquet"
df_spark = spark.read.parquet(gcs_path)
df = df_spark.toPandas()

df['month'] = pd.to_datetime(df['month'])
df['year_month'] = df['month'].dt.to_period('M')
df['psm'] = df['resale_price'] / df['floor_area_sqm']

agg_df = df.groupby(['year_month', 'town']).agg({
    'resale_price': 'mean',
    'floor_area_sqm': 'mean',
    'flat_age': 'mean',
    'storey_median': 'mean',
    'psm': 'mean',
}).reset_index()

agg_df['year_month'] = agg_df['year_month'].dt.to_timestamp()
agg_df = agg_df.sort_values(['town', 'year_month'])

# Create LAG
for lag in [1, 2, 3]:
    agg_df[f'lag_{lag}'] = agg_df.groupby('town')['resale_price'].shift(lag)

# Time feature
agg_df['month_num'] = agg_df['year_month'].dt.month
agg_df['year'] = agg_df['year_month'].dt.year

# Encode town
le = LabelEncoder()
agg_df['town_index'] = le.fit_transform(agg_df['town'])

agg_df = agg_df.dropna()

feature_cols = [
    'lag_1', 'lag_2', 'lag_3',
    'floor_area_sqm', 'flat_age', 'storey_median',
    'psm',
    'month_num', 'year', 'town_index'
]
X = agg_df[feature_cols]
y = agg_df['resale_price']

# Get dataset
train_size = int(len(agg_df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train XGBoost
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)
model.save_model("gs://hrpaps-bucket/models/xgboost_resale_model_mixed.json")

# Test prediction
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

metrics_df = pd.DataFrame([{
    "RMSE": rmse,
    "MAE": mae,
    "R2_Score": r2,
    "MAPE (%)": mape
}])

pred_vs_actual = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

spark_metrics_df = spark.createDataFrame(metrics_df)
spark_pred_vs_actual = spark.createDataFrame(pred_vs_actual)
spark_metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://hrpaps-bucket/logs/apl_model_metrics.csv")
spark_pred_vs_actual.coalesce(1).write.mode("overwrite").option("header", "true").csv("gs://hrpaps-bucket/logs/apl_predictions_vs_actual.csv")

spark.stop()
