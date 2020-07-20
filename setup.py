import setuptools


with open("README.md", "r") as fh:
  long_description = fh.read()


setuptools.setup(
  name="hie",
  version="0.1.3",
  author="sherk",
  author_email="sherkfung@gmail.com",
  description="A heritance of COCO-api with binary matching",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Zepyhrus/hie",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)