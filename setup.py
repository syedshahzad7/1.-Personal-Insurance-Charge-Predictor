from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function reads all the libraries mentioned in the requirements.txt file one by one and then puts each of these libraries into a list and returns that list. This list is actually used by the install_requires in the setup, which downloads all the libraries listed in the list it is given.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
name = 'Personal-Insurance-Charge-Predictor',
version = '0.0.1',
author = 'Shahzad',
author_email = 'shahzadksa777@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')

)