import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines())

setuptools.setup(
    name="TFNetworks",
    version="0.1",
    author="Mathieu Tuli",
    author_email="tuli.mathieu@gmail.com",
    description="TFNetworks is an easy to use API for building various types of networks in TensorFlow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MathieuTuli/TFNetworks",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
    classifiers=[
        "Intened Audience :: Anyone",
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietar License",
    ],
    scripts=glob('bin/*'),
)
