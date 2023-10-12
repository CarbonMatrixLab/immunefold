from setuptools import setup, find_packages

setup(
    name = 'AbFold',
    version = '2.0.0',
    license='MIT',
    description = 'AbFold - AI based methods for antibody structure prediction',
    author = 'Haicang Zhang',
    author_email = 'zhanghaicang@gmail.com',
    
    packages=find_packages(include=['abfold', 'abfold/*']),
    include_package_data=True,
    package_data={
        'abfold': ['common/default_rigids.json', 'common/stereo_chemical_props.txt'],
    },
  
  install_requires=[],
)
