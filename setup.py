from setuptools import setup, find_packages

def read_requirements():
    """
    Reads the requirements from the requirements.txt file.
    """
    with open('requirements.txt', 'r') as req:
        return [line.strip() for line in req if not line.startswith('#')]

setup(
    name='rv-simulator',
    version='1.0.0',
    author='Go-Spect',
    author_email='your_email@example.com',
    description='A synthetic radial velocity (RV) data generator for multi-planet systems.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Go-Spect/inject_recovery_keplerian',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Ou a licença que você preferir
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'run-rv-sim = rv_simulator.cli:main',
        ],
    },
)