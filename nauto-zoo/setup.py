from pathlib import Path
from subprocess import check_output

from setuptools import find_packages, setup


def get_version() -> str:
    init_file = Path(__file__).parent / 'nauto_zoo' / '__version__.py'
    with open(init_file, 'r') as f:
        values = {}
        exec(f.read(), values)
        if '__version__' not in values:
            raise RuntimeError('__version__ could not be found')

        branch_name = check_output(
            "git branch | grep \* | cut -d ' ' -f2", shell=True
        ).decode().rstrip()

        version = values['__version__']
        if branch_name != 'master':
            version = '.'.join([version, 'dev'])

        return version

setup(
    name='nauto-zoo',
    version=get_version(),
    author='nauto.com',
    author_email='',
    license='',
    python_requires='>=3.7.0',
    packages=find_packages(
        exclude=['tests']
    ),
    include_package_data=True,
    install_requires=[
        'protobuf==3.20.0',
        #'nauto_datasets>=0.9.5',
        'numpy>=1.16.4',
        'joblib==1.0.1',
        'scikit-learn==0.24.2',
        'xgboost==1.4.1'
    ],
    extras_require={
        'dev': [
            'pandas>=0.25.3',
            'scipy>=1.3.0',
        ],
        'preprocess': [
            'h5py>=3.6.0',
            'moviepy==1.0.1',
            'tensorflow==2.8.0',
            'tensorflow-addons',
            'opencv-python-headless==4.1.2.30'
        ],
        'test': ['pytest==5.2.4']
    },
    scripts=[])

