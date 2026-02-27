from setuptools import find_packages , setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will the lisst of requirements 
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    return requirements
    
setup(
    name='angraj',
    version='0.0.1',
    author='angraj',
    author_email='angraj@11011',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )

