import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'

            #load the model file
            model = load_object(file_path=model_path)

            #load the preprocessor file
            preprocessor = load_object(file_path=preprocessor_path)

            #standard scaling the features
            data_scaled = preprocessor.transform(features)

            #Make prediction
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Private_Attorney:int,	
                 Marital_Status:int,
                 Specialty:str,
                 Insurance:str,
                 Gender:str,
                 Age_Group,
                 Severity_Category,
                 Attorney_Severity:int):
        
        self.Private_Attorney = Private_Attorney
        self.Marital_Status = Marital_Status
        self.Specialty = Specialty
        self.Insurance = Insurance
        self.Gender = Gender
        self.Age_Group = Age_Group
        self.Severity_Category = Severity_Category
        self.Attorney_Severity = Attorney_Severity

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Private_Attorney": [self.Private_Attorney],
                "Marital_Status" : [self.Marital_Status],
                "Specialty" : [self.Specialty],
                "Insurance" : [self.Insurance],
                "Gender" : [self.Gender],
                "Age_Group" : [self.Age_Group],
                "Severity_Category" : [self.Severity_Category],
                "Attorney_Severity" : [self.Attorney_Severity]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

        