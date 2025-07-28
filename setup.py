from setuptools import setup, find_packages

setup(
    name="gpr_reservoir",
    version="0.1.0",
    description="Gaussian Process Regression and Kriging for 3D Reservoir Simulation Data",
    author="Kyle T. Jones",
    author_email="kyletjones@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0",
        "gpytorch>=1.11",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.2",
        "pykrige>=1.7",
        "matplotlib>=3.7",
        "joblib>=1.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
