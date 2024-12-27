from setuptools import setup, find_packages

setup(
    name='gp_auto',
    version='1.0.0',
    author='Saurabh Deshpande',
    description='Autoencoder + GP framework package for probabilistic regression',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/saurabhdeshpande93/gp-auto-regression/tree/main',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3.10',
)