

"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
import pickle
import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")


    # Read the downloaded CSV file
    dataset = pd.read_csv(fn)

    dataset["new_rating"] = dataset["rating"].map({4.0:"positive", 5.0:"positive", 1.0:"negative", 2.0:"negative", 3.0:"negative"})
        
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    dataset["new_rating"] = encoder.fit_transform(dataset["new_rating"])
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    dataset[dataset.columns[dataset.dtypes!=object][1:]] = scaler.fit_transform(dataset[dataset.columns[dataset.dtypes!=object][1:]])
    
    x = dataset[dataset.columns[2:38]]
    y = dataset.new_rating
    
    print(x.columns)
    
    logger.info("going into train test split")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4)


    trans = {
        'One_Hot': encoder,
        'scaler': scaler,
    }   

    # Split the data
    pd.DataFrame(np.c_[y_train, X_train]).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_val, X_val]).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_test, X_test]).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

