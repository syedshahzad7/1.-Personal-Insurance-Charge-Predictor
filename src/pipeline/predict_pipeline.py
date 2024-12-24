import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):     #This is where the model makes prediction
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            #Load the model
            model = load_object(file_path = model_path)

            #Load the preprocessor
            preprocessor = load_object(file_path = preprocessor_path)

            #Scale the data in the same way it was done during training
            data_scaled = preprocessor.transform(features)

            #Finally, the model will predict based on the data
            preds = model.predict(data_scaled)                  #I think this predict function returns a list, even if we are giving it only one row of data to predict. So preds is a list
            return preds
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:           #This class will be responsible for mapping the inputs given by the user in the webpage to the backend
    def __init__(self,
                 sex: str,
                 smokes: str,
                 region: str,
                 age: int,
                 bmi: float,
                 children: int):
        self.sex = sex
        self.smokes = smokes
        self.region = region
        self.age = age
        self.bmi = bmi
        self.children = children

    #Now, we create a function that take these values and converts it into a dataframe, because our model was trained on a dataframe
    def get_data_as_data_frame(self):
        try:
            #To convert into dataframe, you first need to convert it into a dictionary
            custom_data_input_dict = {
                "sex": [self.sex],
                "smokes":[self.smokes],
                "region": [self.region],
                "age": [self.age],
                "bmi": [self.bmi],
                "children": [self.children]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)