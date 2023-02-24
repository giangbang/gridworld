from setuptools import setup, find_packages
setup(
    name = "gridworld",
    version = "0.0.1",
    description = ("gridworld environment for RL"),
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib"
    ],
    package_data={'': ['*.txt']},
    include_package_data=True,
)
