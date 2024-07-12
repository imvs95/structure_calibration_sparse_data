import pathlib
from setuptools import setup, find_packages  # type: ignore

# The directory containing this file
HERE = pathlib.Path(__file__).parent
REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(  # type: ignore
    name="celibration",
    version="0.0.1",
    description=open("README.md").read(),
    url="https://github.com/riwish/kalibratietechnieken",
    author="NPAI",
    packages=find_packages(
        exclude=(
            "tests",
            "police_simulation_model",
            "pydsol",
            "overrides",
        )
    ),
    include_package_data=True,
    entry_points={
        "console_scripts": ["celibration = celibration.entrypoint:main"]  # type: ignore
    },
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
)
