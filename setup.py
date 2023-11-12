from setuptools import setup


install_requires = []
with open("requirements.txt", "r") as f:
    lines = f.readlines()
    install_requires = [line.rstrip() for line in lines]

setup(
    name="F16model",
    author="lalapopa",
    version="0.0.1",
    description='Module for evaluatin F16 env',
    long_description='TODO: this description',
    packages=['F16model'],
    install_requires=install_requires,
)

