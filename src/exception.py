import sys                #Any exception that gets detected, that exception's information is available within the 'sys' library. What type of exception has occured will be explained to us by this sys library. Using this, we can also traceback where the exception has occured
import logging 

def error_message_detail(error, error_detail:sys):              #Whenever an exception gets raised, this is the custom error message that I want to be pushed on my console
              _, _, exc_tb = error_detail.exc_info()            #error_detail is an instance of sys. sys has an attribute exc_info that returns three pieces of information: exception type, exception value and the point in the program where the exception occured. Not interested in the first 2 things, and only saving the third info in the exc_tb
              file_name = exc_tb.tb_frame.f_code.co_filename    #To fetch the file name where the exception has been caught
              error_message = "Error occured: python script (file) name: [{0}], at line number: [{1}], error message: [{2}]".format(file_name, exc_tb.tb_lineno, str(error))

              return error_message

class CustomException(Exception):
        def __init__(self, error_message, error_detail:sys):
                super().__init__(error_message)
                self.error_message = error_message_detail(error_message, error_detail = error_detail)

        def __str__(self):
                return self.error_message
 


