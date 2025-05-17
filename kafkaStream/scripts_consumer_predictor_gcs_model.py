from kafka import KafkaConsumer
import json
import pandas as pd
import xgboost as xgb
from google.cloud import storage
from datetime import datetime
import os
import time

KAFKA_BROKER = '35.194.211.144:9092'
TOPIC = 'resale_stream'
GROUP_ID = 'resale-predict-group'

GCS_MODEL_PATH = "models/xgboost_resale_model_mixed.json"
BUCKET_NAME = 'hrpaps-bucket'
GCS_PREDICT_FILE = 'predictions/prediction_results.csv'
LOCAL_MODEL_FILE = '/tmp/xgboost_model.json'
LOCAL_PREDICT_FILE = '/tmp/prediction_results.csv'

feature_cols = [
    'lag_1', 'lag_2', 'lag_3',
    'floor_area_sqm', 'flat_age', 'storey_median',
    'psm', 'month_num', 'year', 'town_index'
]

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
model_blob = bucket.blob(GCS_MODEL_PATH)

if not model_blob.exists():
    raise FileNotFoundError("model not existing")

model_blob.download_to_filename(LOCAL_MODEL_FILE)

model = xgb.XGBRegressor()
model.load_model(LOCAL_MODEL_FILE)

csv_blob = bucket.blob(GCS_PREDICT_FILE)
if csv_blob.exists():
    csv_blob.download_to_filename(LOCAL_PREDICT_FILE)
    print("已下载历史预测文件:", LOCAL_PREDICT_FILE)
else:
    pd.DataFrame(columns=["timestamp", "town"] + feature_cols + ["resale_price", "predicted_resale_price"]).to_csv(LOCAL_PREDICT_FILE, index=False)
    print("已创建空预测文件")


consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id=GROUP_ID,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Kafka Consumer start, waiting for messages...")


results_buffer = []
last_upload_time = time.time()
UPLOAD_INTERVAL = 10  

for message in consumer:
    msg = message.value
    try:
        X_pred = pd.DataFrame([msg])[feature_cols]
        y_pred = model.predict(X_pred)[0]

        result = {
            "timestamp": msg["timestamp"],
            "town": msg["town"],
            **{col: msg[col] for col in feature_cols},
            "resale_price": msg.get("resale_price"),
            "predicted_resale_price": round(y_pred, 2)
        }

        results_buffer.append(result)
        print("Predicted:", result)

        if time.time() - last_upload_time >= UPLOAD_INTERVAL:
            if results_buffer:
                df_new = pd.DataFrame(results_buffer)
                df_new.to_csv(LOCAL_PREDICT_FILE, mode='a', header=False, index=False)
                csv_blob.upload_from_filename(LOCAL_PREDICT_FILE)
                print(f"Uploaded {len(results_buffer)} predictions to GCS.")
                results_buffer.clear()
                last_upload_time = time.time()

    except Exception as e:
        print("error:", e)
