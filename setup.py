import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hyperspex", # Replace with your own username
    version="0.0.2",
    author="Adrien Rousseau",
    author_email="adrienrousseau@gmail.com",
    description="This package allow you to explore hyperspectral data easly.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdrienR09/hyperspex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)