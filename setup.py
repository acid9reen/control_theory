from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="control_theory",
    install_requires=[
        "scipy == 1.8.0",
        "numpy == 1.22.2",
        "matplotlib == 3.5.1",
        "notebook == 6.4.8",
    ],
)