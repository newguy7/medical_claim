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

from src.utils import save_object


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
            numerical_columns = ['Private Attorney', 'Marital Status', 'Severity','Attorney_Severity']
            categorical_columns = ['Specialty', 'Insurance', 'Gender', 'Age Group']

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
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Applying feature engineering...")

            # create Age Group
            train_df['Age Group'] = pd.cut(train_df['Age'], bins=[0,20,40,60,80,100],labels=['0-20', '21-40', '41-60', '61-80', '80+'])
            test_df['Age Group'] = pd.cut(test_df['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '80+'])            

            # Create Interaction Term
            train_df['Attorney_Severity'] = train_df['Private Attorney'] * train_df['Severity']
            test_df['Attorney_Severity'] = test_df['Private Attorney'] * test_df['Severity']

            logging.info("Feature engineering completed.")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj()
            

            target_column_name = 'Amount'            
          

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]           
            

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")


            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)   

            # Convert sparse matrix to dense NumPy array
            input_feature_train_array = input_feature_train_array.toarray()

            # Ensure target array is 2D            
            target_feature_train_array = target_feature_train_df.to_numpy().reshape(-1, 1) #added         

            # Convert sparse matrix to dense NumPy array
            input_feature_test_array = input_feature_test_array.toarray()

            # Ensure target array is 2D            
            target_feature_test_array = target_feature_test_df.to_numpy().reshape(-1, 1) #added 

            # Ensure both arrays have matching rows
            assert input_feature_train_array.shape[0] == target_feature_train_array.shape[0], \
                "Mismatch in number of rows between input features and target feature."
            
            # # Ensure input feature array is 2D
            # print("input_feature_train_array shape:", input_feature_train_array.shape)

            # # Ensure target feature array is 2D
            # print("target_feature_train_array shape:", target_feature_train_array.shape)

            # # Ensure the arrays are NumPy arrays
            # print("Type of input_feature_train_array:", type(input_feature_train_array))
            # print("Type of target_feature_train_array:", type(target_feature_train_array))
            
            # Concatenate input features and target
            # train_arr = np.concatenate([input_feature_train_array, target_feature_train_array], axis=1)

            # test_arr = np.concatenate([input_feature_test_array, target_feature_test_array], axis=1)

            train_arr = np.c_[
                input_feature_train_array, target_feature_train_array
            ]          


            test_arr = np.c_[
                input_feature_test_array, target_feature_test_array
            ]        
            

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


