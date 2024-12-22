#In this module, we read the dataset from some external source and split it into training and testing dataset. The module should take three inputs: path where the training dataset needs to be stored in the project folder, path where the testing dataset needs to be stored in the project folder, and path where the raw dataset needs to be stored in the project folder. It then creates these paths and files properly, splits the dataset and returns the paths where training and testing datasets are stored.
#The paths of the training and testing dataset are then passed as inputs to the next stage i.e., the transformation stage

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

#defining a class for inputs to the ingestion module
@dataclass                      # -> @dataclass decorator is a shortcut to make a class that just stores some information. It eliminates the need to create a constructor for your class. You can directly create your class's attributes without having to define a constructor (def __init__)
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")   #first attribute of this class that stores the path of training data
    test_data_path: str = os.path.join('artifacts', "test.csv")     #second attribute of this class that stores the path of testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")      #third attribute of this class that stores the path of raw data
#So in the data ingestion module, we'll probably give three inputs. The raw data which has to be splitted into training and testing dataset, and the paths were the splitted data needs to be stored. 


#Now we start our actual class for this module
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()    #When we create an object for this DataIngestion class, we need to pass the three paths mentioned above. So this was a very indirect way of creating attributes for the DataIngesion class. The three paths, namely: train_data_path, test_data_path and raw_data_path gets stored in this single attribute: self.ingestion_config . So inside this variable, there are three sub objects which are those 3 paths

    def initiate_data_ingestion(self):          #custom function to actually initialize the data ingestion process
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook\data\insurance.csv")        #first of all, we read the dataset. If you had to read from some database, say mongodb, you could call the function that reads the data from mongodb here. That function which reads data from mongodb would be defined by you in the utils package.
            logging.info("Read the data and stored in the dataframe")

            #Now, we create the artifacts folder if it already doesn't exist (if it exists, we ignore it). The os.path.dirname function fetches only the folder name (artifacts) in artifacts/train_data_path (which is the value of self.ingestion_config.train_data_path). Thus, this line creates the artifacts folder.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)      

            #Now, we'll store the data in the raw_data_path defined earlier
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Raw data stored in the artifacts folder")

            logging.info("Train test split initiated.")
            train_dataset, test_dataset = train_test_split(df, test_size = 0.2, random_state = 42)

            train_dataset.to_csv(self.ingestion_config.train_data_path, index = False, header = True) #storing training dataset in 'train_data_path' path
            test_dataset.to_csv(self.ingestion_config.test_data_path, index = False, header = True)    #storing testing dataset in 'test_data_path' path

            logging.info("Ingestion of data completed!")

            #Now finally, we have to return the output of the ingestion module
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            #These two are the outputs from the ingestion module that shall be passed on to the next step which is the data transformation.
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

        










