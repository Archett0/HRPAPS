import pandas as pd
import numpy as np
from kafka import KafkaProducer
import json
import time
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from google.cloud import storage

KAFKA_BROKER = '35.194.211.144:9092'
TOPIC = 'resale_stream'
GCS_MODEL_PATH = "models/xgboost_resale_model_mixed.json"
BUCKET_NAME = 'hrpaps-bucket'
LOCAL_MODEL_FILE = '/tmp/xgb_model.json'

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(GCS_MODEL_PATH)
blob.download_to_filename(LOCAL_MODEL_FILE)

model = xgb.XGBRegressor()
model.load_model(LOCAL_MODEL_FILE)

df = pd.read_parquet(
    "gs://hrpaps-bucket/test/forMachineLearning.parquet",
    engine="pyarrow",
    storage_options={"token": "cloud"}
)
df['month'] = pd.to_datetime(df['month'])
df['year_month'] = df['month'].dt.to_period('M')
df['psm'] = df['resale_price'] / df['floor_area_sqm']

le = LabelEncoder()
df['town_index'] = le.fit_transform(df['town'])
town_index_map = dict(zip(le.classes_, le.transform(le.classes_)))

agg_df = df.groupby(['year_month', 'town']).agg({'resale_price': 'mean'}).reset_index()
agg_df['year_month'] = agg_df['year_month'].dt.to_timestamp()
agg_df = agg_df.sort_values(['town', 'year_month'])

lag_dict = {}
for town in agg_df['town'].unique():
    recent = agg_df[agg_df['town'] == town].tail(3)
    if len(recent) == 3:
        lag_dict[town] = [
            recent.iloc[-1]['resale_price'],
            recent.iloc[-2]['resale_price'],
            recent.iloc[-3]['resale_price']
        ]

last_month = df['month'].max().to_period('M')
recent_df = df[df['month'].dt.to_period('M') == last_month]

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Kafka Producer started. Streaming 1 valid sample every 1s...")

feature_cols = [
    'lag_1', 'lag_2', 'lag_3',
    'floor_area_sqm', 'flat_age', 'storey_median',
    'psm', 'month_num', 'year', 'town_index'
]

while True:
    now = datetime.now()
    month_num = now.month
    year = now.year
    today_str = now.strftime("%Y-%m-%d")

    row = recent_df.sample(1).iloc[0]
    town = row['town']
    if town not in lag_dict:
        continue

    sample = {
        "timestamp": today_str,  
        "town": town,
        "lag_1": lag_dict[town][0],
        "lag_2": lag_dict[town][1],
        "lag_3": lag_dict[town][2],
        "floor_area_sqm": row['floor_area_sqm'],
        "flat_age": row['flat_age'],
        "storey_median": row['storey_median'],
        "psm": row['psm'],
        "month_num": month_num,
        "year": year,
        "town_index": town_index_map[town]
    }

    X_pred = pd.DataFrame([sample])[feature_cols]
    y_pred = model.predict(X_pred)[0]
    simulated_price = round(y_pred * (1 + np.random.normal(0, 0.05)), 2)

    sample["resale_price"] = simulated_price

    clean_data = {k: (v.item() if isinstance(v, (np.generic,)) else v) for k, v in sample.items()}
    producer.send(TOPIC, value=clean_data)
    producer.flush()
    print("Sent:", clean_data)
    time.sleep(1)
