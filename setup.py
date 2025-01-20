from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

req_path = here / "requirements.txt"
with open(req_path) as f:
    requirements = f.read().splitlines()

setup(
    name="",  # Required
    version="0.0.1",  # Required
    description="",  # Optional
    long_description=long_description,  # Optional
    packages=find_packages(),  # Required
    python_requires=">=3.8, <4",
    install_requires=requirements,
    dependency_links=["git+https://github.com/..."],
    extras_require={  # Optional
        "dev": ["pytest", "pytest-cov"],
    },
)

# # List additional groups of dependencies here (e.g. development
# # dependencies). Users will be able to install these using the "extras"
# # syntax, for example:
# #
# #   $ pip install sampleproject[dev]
# #
# # Similar to `install_requires` above, these must be valid existing
# # projects.
# extras_require={  # Optional
#     "dev": ["check-manifest"],
#     "test": ["coverage"],
# },

# # Entry points. The following would provide a command called `sample` which
# # executes the function `main` from this package when invoked:
# entry_points={  # Optional
#     "console_scripts": [
#         "sample=sample:main",
#     ],
# },
# List additional URLs that are relevant to your project as a dict.
#
# # This field corresponds to the "Project-URL" metadata fields:
# # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
# #
# # Examples listed include a pattern for specifying where the package tracks
# # issues, where the source is hosted, where to say thanks to the package
# # maintainers, and where to support the project financially. The key is
# # what's used to render the link text on PyPI.
# project_urls={  # Optional
#     "Bug Reports": "https://github.com/pypa/sampleproject/issues",
#     "Funding": "https://donate.pypi.org",
#     "Say Thanks!": "http://saythanks.io/to/example",
#     "Source": "https://github.com/pypa/sampleproject/",
# },
# )
