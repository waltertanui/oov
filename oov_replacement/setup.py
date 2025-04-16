from setuptools import setup, find_packages

setup(
    name="oov_replacement",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.6.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for replacing out-of-vocabulary (OOV) words with in-vocabulary (IV) words",
    keywords="nlp, oov, text processing",
    python_requires=">=3.6",
)