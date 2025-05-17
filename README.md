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



### 3.kafka stream run commands
based on gcp kafka cluster and model, and should upload procuder and consumer to gcp also
#### 3.1 pruducer run commands on gcp shell
gsutil cp gs://hrpaps-bucket/scripts/consumer_predictor_gcs_model.py .
pip install kafka-python pandas xgboost google-cloud-storage
python3 consumer_predictor_gcs_model.py
#### 3.2 consumer run commands on gcp shell
gsutil cp gs://hrpaps-bucket/scripts/consumer_predictor_gcs_model.py .
pip install kafka-python pandas xgboost google-cloud-storage
python3 consumer_predictor_gcs_model.py




### 5.Contract
github: https://github.com/Archett0/HRPAPS
email：