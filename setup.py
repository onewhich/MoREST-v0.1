from setuptools import setup, find_packages

setup(
    name="MoREST",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy","scipy","ase","tqdm","scikit-learn","MoREAT"],
)
