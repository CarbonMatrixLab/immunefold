from setuptools import setup, find_packages

setup(
  name = 'AbFold',
  version = '1.0.0',
  license='MIT',
  description = 'AbFold - AI based methods for antibody structure prediction',
  author = 'Haicang Zhang',
  author_email = 'zhanghaicang@gmail.com',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'protein folding'
  ],
  
  packages=['abfold'],
  package_dir={'abfold': './abfold'},
  package_data={'abfold': ['./abfold/common/new_rigid_schema.json']},
  
  install_requires=[],
  test_suite = 'tests',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
