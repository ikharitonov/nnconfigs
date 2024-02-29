from setuptools import setup, find_packages

setup(
    name='nnconfigs',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.2',
        'pandas',
        'tqdm'
    ]
)