from setuptools import setup, find_packages

setup(
    name="loco-worldline",
    version="0.1.0",
    description="Lorentzian Worldline Attention for Hierarchical Perception",
    author="Anonymous",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
)

# NOTE: This setup.py is incomplete and may not work correctly.
# Use requirements_frozen_cluster.txt for installation instead.
