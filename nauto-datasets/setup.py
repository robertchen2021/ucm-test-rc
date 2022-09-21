from pathlib import Path
from subprocess import check_output

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def build_protos():
    from grpc.tools import command
    command.build_package_protos('.')


class BuildPackageProtos(build_py):
    def run(self):
        build_protos()
        build_py.run(self)


class BuildPackageDevelopProtos(develop):
    def run(self):
        build_protos()
        develop.run(self)


def get_version() -> str:
    init_file = Path(__file__).parent / 'nauto_datasets' / '__init__.py'
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


test_extra = [
    'flask==1.1.2',
    'moto==1.3.16',
    'pytest>=5.0.1',
    'pytest-asyncio>=0.10.0',
    'requests>=2.22.0'
]
setup(
    name='nauto-datasets',
    version=get_version(),
    author='nauto.com',
    author_email='marcin.zieminski@nauto.com',
    license='',
    python_requires='>=3.6.0',
    packages=find_packages(
        exclude=['integration_tests', 'tests', 'scripts', '.deps']
    ),
    include_package_data=True,
    install_requires=[
        'aiobotocore[boto3]==0.11.0',
        'click==7.0',
        'ffmpeg-python==0.2.0',
        'grpcio>=1.12.1',
        'grpcio-tools>=1.12.1',
        'matplotlib==3.1.2',
        'numpy>=1.14.5',
        'pandas>=0.23.3',
        'plotly==4.4.1',
        'protobuf-to-dict>=0.1.0',
        'pyarrow>=0.10.0',
        'pyfiglet==0.7',
        'python-dateutil>=2.8.0',
        'qds_sdk==1.13.2',
        'grpcio>=1.12.1',
        'grpcio-tools>=1.12.1',
        'typing_inspect==0.4.0'
    ],
    extras_require={
        'tf-cpu': 'tensorflow>=1.14.0',
        'tf-gpu': 'tensorflow-gpu>=1.14.0',
        'spark': 'pyspark==2.4.0',
        'dev': test_extra
    },
    setup_requires=[
        'grpcio>=1.12.1',
        'grpcio-tools>=1.12.1'
    ],
    tests_require=test_extra,
    scripts=[],
    cmdclass={
        'build_py': BuildPackageProtos,
        'develop': BuildPackageDevelopProtos
    },
    entry_points={
        'console_scripts': ['nauto-datasets=nauto_datasets.cli:main']
    }
)
