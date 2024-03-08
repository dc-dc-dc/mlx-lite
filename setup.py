from setuptools import setup, find_packages

setup(
    name="mlxlite",
    version="0.0.1",
    author="Diogo Da Cruz",
    python_requires=">=3.8",
    packages=find_packages(where="mlxlite"),
    install_requires=["mlx", "flatbuffers"],
    include_package_data=True
)
