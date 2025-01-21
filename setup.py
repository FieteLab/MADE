from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

requirements = [
    "numpy",
    "matplotlib",
    "scikit-learn",
    "loguru",
    "pre-commit",
    "black",
    "flake8",
]

setup(
    name="ManifoldAttractors",  # Required
    version="0.0.1",  # Required
    description="Manifold Activity Direct Embedding (MADE) - A toolkit for engineering Continuous Attractor Networks.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    author="Federico Barabas",
    author_email="claudif@mit.com",  # You should update this
    url="https://github.com/FieteLab/MADE",
    packages=find_packages(),  # Required
    python_requires=">=3.8, <4",
    install_requires=requirements,
    dependency_links=[""],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="neural-dynamics, manifold, attractor-networks",
)
