"""
Setuptools for devops cloudformation scripts.
Template copied from https://github.com/pypa/sampleproject.
"""
from setuptools import setup, find_packages
from subprocess import check_output
from codecs import open
from pathlib import Path


def get_version() -> str:
    init_file = Path(__file__).parent / 'nauto_ds_utils' / '__init__.py'
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
            nums = version.split('.')
            version = '.'.join([nums[0], nums[1], 'dev' + nums[2]])

        return version


# Get the long description from README file
with open('README.md', encoding='utf-8') as inf:
    long_description = inf.read()

setup(
    name='nauto_ds_utils',
    version=get_version(),
    author='nauto.com',
    author_email='oscar.diec@nauto.com',
    description='Nauto utils for working with data in PySpark environment',
    long_description=long_description,
    url='https://github.com/nauto/nauto-ai/tree/master/nauto-ds-utils/sensor_preprocessing/readme.md',
    packages=find_packages(exclude=['docs', 'tests']),
    # This is subject ot change as nauto-datasets gets updates
    install_requires=[
        'nauto-datasets>=0.9.0',
        'pytest>=6.2.5',
    ]
)
