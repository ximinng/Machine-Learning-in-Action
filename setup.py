# -*- coding: utf-8 -*-
"""
   Description :   Package the project
   Author :        xxm
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setup(name='joblib',
          version='0.1',
          author='ximing Xing',
          author_email='ximingxing@gmail.com',
          url='https://ximingxing.github.io/',
          description='Machine Learning in Action',
          long_description=long_description,
          license='BSD',
          classifiers=[
              # 3 - Alpha
              # 4 - Beta
              # 5 - Production/Stable
              'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          python_requires='>=3',
          install_requires=['numpy>=1.17.2',
                            'scikit-learn>=0.21.2',
                            'loguru>=0.3.2',
                            'scipy>=1.3.1', 'scrapy'],
          platforms='any',
          packages=find_packages(exclude=['/data', '*.pkz'])
          )
