from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="timeseries_agent",
    version="0.0.23", 
    author="Collins Patrick Ohagwu",            # Replace with your name
    author_email="cpohagwu@gmail.com", # Replace with your email
    description="A Policy Gradient RL agent for time series prediction using PyTorch Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cpohagwu/timeseries_agent", # Optional: Replace with your repo URL
    packages=find_packages(), 
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "lightning>=2.5.0",    
        "matplotlib>=3.0.0",
        'torch>=1.9.0; platform_system=="Linux"',
        'torch>=1.9.0; platform_system=="Darwin"',
        'torch>=1.9.0; platform_system=="Windows"',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Specify minimum Python version
)
