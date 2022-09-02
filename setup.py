from setuptools import setup, Extension, find_packages
from codecs import open
from os import path
import warnings

package_name = 'istat'
example_dir = 'examples/'
bin_dir = 'bin/'
example_data_dir = example_dir + 'examples_data/'

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=package_name,
    author='zed.uchicago.edu',
    author_email='ishanu@uchicago.edu',
    version = str(version['__version__']),
    packages=find_packages(),
    scripts=[],
    url='https://github.com/zeroknowledgediscovery/emergenet',
    license='LICENSE',
    description='Superfast Risk Estimation of Emerging Pathogens',
    keywords=[
        'computational biology',
        'decision trees', 
        'machine learning', 
        'emerging pathogens'],
    download_url='https://github.com/zeroknowledgediscovery/emergenet/archive/'+str(version['__version__'])+'.tar.gz',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "scikit-learn", 
        "scipy", 
        "numpy",  
        "pandas",
        "joblib",
        "quasinet",
        "scipy"],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3.6"],
    include_package_data=True,
    )
