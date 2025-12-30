from setuptools import find_packages , setup 
from typing import List


def get_requirements(file_path :str ) ->List[str]:
    '''
    this function will return list or requirements
    '''
    requirements = [] 
    with open(file_path ) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n" , "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements




    


setup(name = "mlproject",
      version =  '0.0.1',
      author = "bhargav",
      author_email = 'bhargavjadav78@gmail.com',
      packages= find_packages(),
      install_requires = get_requirements('requirements.txt') 
    )