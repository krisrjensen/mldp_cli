from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mldp_cli",
    version="0.1.0",
    author="Kristophor Jensen",
    description="Master CLI for orchestrating MLDP tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krisrjensen/mldp_cli",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "psycopg2-binary>=2.9.0",
        "tabulate>=0.9.0",
        "prompt-toolkit>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mldp=mldp:main",
            "mldp-cli=cli:cli",
            "mldp-shell=mldp_shell:main",
            "mldp-basic-shell=interactive_cli:main",
        ],
    },
)