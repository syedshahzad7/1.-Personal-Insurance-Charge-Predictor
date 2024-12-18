#Any execution that happens, we should log all those execution information in files to track errors.

import logging
import os

from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"   #This will be the name of the log text files created.
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)          #Path where the log files will be saved. This is set to the Current Working Directory which is the project folder. The files will be saved as logs followed by the date and time convention mentioned by the LOG_FILE
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



