import pathlib
from setuptools import find_packages, setup


def read_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file.readlines() if not line.startswith("#")]


# read requirements from requirements.txt
reqs = read_requirements("./requirements.txt")

setup(
    name="moondream",
    version="0.0.1", 
    description="a python package for loading images",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    Homepage="https://github.com/vikhyat/moondream",
    url="https://github.com/vikhyat/moondream",
    Issues="https://github.com/vikhyat/moondream/issues",
    authors=[{"name": "Vik Korrapati", "email": "vikhyatk@gmail.com"}],
    author_email="vikhyatk@gmail.com",
    license="Apache 2.0 License",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    classifiers=["Topic :: Utilities", "Programming Language :: Python :: 3.9"],
    # requires=["setuptools", "wheel", "typing", "pillow", "numpy", "requests"],
    install_requires= reqs
)