import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorflow-networks",
    version="0.0.1",
    author="Mathieu Tuli",
    author_email="tuli.mathieu@gmail.com",
    description="Various machine learning networks built in TensorFlow, adaptable to configuration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
