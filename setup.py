from setuptools import setup, find_packages

# Read requirements.txt and exclude lines starting with -e or --
def parse_requirements(filename):
    with open(filename) as f:
        lines = f.readlines()
    # Filter out empty lines and lines starting with -e or --
    return [line.strip() for line in lines if line.strip() and not line.startswith(('-e', '--'))]

setup(
    name='Market Predictor Algorithm',
    version='0.1',
    description='ML based Market Prediction',
    author='ANP',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=parse_requirements('requirements.txt'),
)
