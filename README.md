# HRPAPS - HDB Resale Price Analysis and Prediction System

### 1. Project Structure

```text
├─.gitignore
├─README.md
├─requirements.txt
├─notebooks
│     ├─00_tmp.ipynb
│     └─01_data_ingestion.ipynb
├─kafkaStream
│     ├─scripts_consumer_predictor_gcs_model.py (consumer)
│     └─scripts_producer_gcs.py (producer)
├─Data_Ingestion_Processing
│     ├─Data_Ingestion.py
│     └─Data_Processing.py
├─Data_Analytics_n_Machine_Learning
│     ├─APL_Generate_Data_n_Predict.py
│     ├─APL_Model.py
│     ├─APL_Model_Check.py
│     ├─APL_TSF_Generate_n_Predict.py
│     └─APL_TSF_Model.py
```
### 2. Batch Run Commands

#### 2.1 Data Ingestion Commands to Run on GCS
```shell
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/Data_Ingestion.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/Data_Processing.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
```

#### 2.2 Regression Model Training to Run on GCS
```shell
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/APL_Model.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/APL_Model_Check.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/APL_Generate_Data_n_Predict.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
```

#### 2.3 Time Series Forecast Model Training to Run on GCS
```shell
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/APL_TSF_Model.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
gcloud dataproc jobs submit pyspark gs://hrpaps-bucket/code/APL_TSF_Generate_n_Predict.py --cluster=dataproc-sg-cluster --region=asia-southeast1 --properties=spark.submit.deployMode=cluster
```

### 3. Kafka Stream Run Commands
based on gcp kafka cluster and model, and should upload procuder and consumer to gcp also

#### 3.1 pruducer run commands on gcp shell
```shell
gsutil cp gs://hrpaps-bucket/scripts/consumer_predictor_gcs_model.py .
pip install kafka-python pandas xgboost google-cloud-storage
python3 consumer_predictor_gcs_model.py
```

#### 3.2 consumer run commands on gcp shell
```shell
gsutil cp gs://hrpaps-bucket/scripts/consumer_predictor_gcs_model.py .
pip install kafka-python pandas xgboost google-cloud-storage
python3 consumer_predictor_gcs_model.py
```

### 5. Contact

- Main Github repo: https://github.com/Archett0/HRPAPS
- Frontend Github repo: https://github.com/Seven0730/HRPAPS
