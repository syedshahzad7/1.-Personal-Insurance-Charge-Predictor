import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


# Just how we created a config class for data ingestion that has the inputs for the ingestion module, we create a config class for transformation module as well
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

#The input given to the transformation module is simply the path where the preprocessor object (once created) needs to be saved. 
# The preprocessor object is basically a pipeline of preprocessing steps like OneHotEncoder and StandardScaler. This preprocessor obj is saved in a pickle file(serialized object easy to save and load it back when needed).
# Later, when you need to process new data (e.g., during model prediction), you can load this preprocessor easily. And to load this new processor, you can easily access it from the 'preprocessor_obj_file_path'


#Now, we implement the actual DataTransformation module
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    #We then create a function that actually does the transformation of the columns. So when we create our preprocessor object (which is obj of this class), we call the object with this function to perform the preprocessing.
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["age", "bmi", "children"]
            categorical_columns = ["sex", "smoker", "region"]

            #we now create the pipeline for dealing with numerical features.
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),      #The pipeline first handles the missing values
                    ("scaler", StandardScaler(with_mean = False))                          #Next, the pipeline applys standardization to each numerical column. This is what the numerical pipeline does
                ]
            )

            #Next, we create the pipeline for categorical features.
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),     #again, the first thing this pipeline handles are the missing values
                    ("one_hot_encoder", OneHotEncoder()),                       #Next, One hot encoding
                    ("scaler", StandardScaler(with_mean = False))                                #Then standardization for categorical columns also. This is not necessary, but we can do it
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #Now, the numerical pipeline and categorical pipeline needs to be combined into a single pipeline.
            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", num_pipeline, numerical_columns),         #In this transformation pipeline, we first do numerical transformation using the numerical pipeline
                ("categorical_pipeline", cat_pipeline, categorical_columns)      #Next we do the categorical transformation using the categorical pipeline

                ]
            )

            #Finally, we return the preprocessor pipeline we created
            return preprocessor            #preprocessor is the preprocessing object. You can use this preprocessor object to fit and transform your dataset.
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    
    #Now we write the function to start the transformation
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)    #First of all, convert those data in dataframes coz our transformations work on dataframes
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Drop duplicate columns from both train and test datasets
            train_df = train_df.loc[:, ~train_df.columns.duplicated()]
            test_df = test_df.loc[:, ~test_df.columns.duplicated()]
            logging.info("Dropped duplicate columns from train and test dataframes")

            logging.info("obtaining preprocessing object")

            #The preprocessor object that we created above, we'll access that as follows
            preprocessing_obj = self.get_data_transformer_object()         #So this is our preprocessing object
            
            target_column_name = "charges"
            numerical_columns = ["age", "bmi", "children"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)   #These are the input features
            target_feature_train_df=train_df[target_column_name]                        #This is the target column

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)     #The fit_transform() method transforms the features in the dataframe into a NumPy array.
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]        #np.c_[] is a shortcut in NumPy to combine two arrays. So here, we are basically combining the preprocessed input features array and np.array(target feature column)
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving the preprocessing object.")

            #Using the utils function to save the preprocessing object as a pkl file. This preprocessing object will be loaded gain for making future predictions
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            #Finally, this module returns the preprocessed training, testing array along with the preprocessor object file path.
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)


    


