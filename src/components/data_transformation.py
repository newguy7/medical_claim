'''
This file is responsible for data preprocessing and feature engineering.
- Creating new features.
- Handling missing values.
- Scaling and encoding features.
'''

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['Amount', 'Severity', 'Age', 'Private Attorney', 'Marital Status']
            categorical_columns = ['Specialty', 'Insurance', 'Gender']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))                    
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            logging.info(f"Numerical columns: {numerical_columns}")

            # combination of numerical and categorical pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transforamtion(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Applying feature engineering...")

            # create Age Group
            train_df['Age Group'] = pd.cut(train_df['Age'], bins=[0.20,40,60,80,100],labels=['0-20', '21-40', '41-60', '61-80', '80+'])
            test_df['Age Group'] = pd.cut(test_df['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '80+'])

            # Create Severity Category
            train_df['Severity Category'] = pd.cut(train_df['Severity'], bins=[1, 3, 6, 9], labels=['Low', 'Medium', 'High'])
            test_df['Severity Category'] = pd.cut(test_df['Severity'], bins=[1, 3, 6, 9], labels=['Low', 'Medium', 'High'])

            # Create Interaction Term
            train_df['Attorney_Severity'] = train_df['Private Attorney'] * train_df['Severity']
            test_df['Attorney_Severity'] = test_df['Private Attorney'] * test_df['Severity']

            logging.info("Feature engineering completed.")

            logging.info("Obtaining preprocessing object")
