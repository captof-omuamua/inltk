import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inltk",
    version="0.9",
    author="Gaurav",
    author_email="contactgauravforwork@gmail.com",
    description="Natural Language Toolkit for Indian Languages (iNLTK)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goru001/inltk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.15",
        "sentencepiece",
    ],
    extras_require={
        # Model-based features (predict_next_words, embeddings, similarity,
        # identify_language) need the original fastai v1 stack, which only
        # installs on older Python versions.
        "models": [
            "fastai==1.0.57",
            "scikit-learn",
            "spacy>=2.0.18",
        ],
    },
)
