from setuptools import setup, find_packages

setup(
    name="alaaraji_segi",
    version="0.1.0",
    description="A library for reading and analyzing SEG-Y files using segyio and obspy.",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "segyio>=1.0.0",
        "obspy>=1.0.0",
        "numpy>=1.18.0",
        "pandas>=1.1.0",
        "matplotlib>=3.2.0",
    ],
)
