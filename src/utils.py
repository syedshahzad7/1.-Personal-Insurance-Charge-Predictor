 #common functionalities that entire project can use

import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill

def save_object(file_path, obj):        #function to save an 'obj' as a pickled file in the mentioned 'file_path'
    try:
        dir_path = os.path.dirname(file_path)   #extract the directory name from the passed file_path
        os.makedirs(dir_path, exist_ok = True)  #Create a directory with that extracted directory's name. If already exists, ignore

        with open(file_path, "wb") as file_obj:     #opening the file specified by the file_path and writing to it in binary mode(wb). You use the file_obj object to write into the file in file_path. 
            dill.dump(obj, file_obj)                #serializes the "obj" into a binary format and write it to the file represented by file_obj (which is basically the file given in the file_path)
    except Exception as e:
        raise CustomException(sys, e)                    
                    