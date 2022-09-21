from pathlib import Path
from subprocess import check_output
from setuptools import find_packages, setup


def get_version() -> str:
    init_file = Path(__file__).parent / 'nauto_ucm_accountant' / '__version__.py'
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
    name='nauto-ucm-accountant',
    version=get_version(),
    author='nauto.com',
    author_email='gleb.sidora@nauto.com',
    license='',
    python_requires='>=3.6.0',
    packages=find_packages(
        exclude=['tests']
    ),
    include_package_data=True,
    extras_require={
        'test': ['pytest==5.3.5'],
        # todo this is to avoid weird conflicts with conda-installed versions on Qubole
        'boto3': ['boto3==1.12.3'],
        'matplotlib': ['matplotlib==3.2.1'],
        'nf': ['nautoflow==0.1.2'],
    },
    scripts=[])
