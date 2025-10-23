#!/usr/bin/env python3
"""
Setup script for Chess Neuro-Symbolic Training
This helps GitHub's dependency graph detect Python dependencies
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chess-neuro-symbolic-training",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid neuro-symbolic approach for chess position analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess-neuro-symbolic-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chess-train=train_chess_improved:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)