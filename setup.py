"""
Setup configuration for Amazon Brand Intelligence Pipeline.

This setup.py enables installation of the package via pip:
    pip install -e .
    pip install -e ".[dev]"
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

# Read dev requirements
dev_requirements_path = Path(__file__).parent / "requirements-dev.txt"
dev_requirements = []
if dev_requirements_path.exists():
    dev_requirements = [
        line.strip()
        for line in dev_requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#") and not line.startswith("-r")
    ]

setup(
    name="amazon-brand-intelligence",
    version="1.0.0",
    author="Amazon Brand Intelligence Team",
    author_email="team@example.com",
    description="AI-powered Amazon brand intelligence and analysis pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/amazon-brand-intelligence",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/amazon-brand-intelligence/issues",
        "Documentation": "https://your-org.github.io/amazon-brand-intelligence/",
        "Source Code": "https://github.com/your-org/amazon-brand-intelligence",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    package_dir={"": "."},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brand-intel=src.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Typing :: Typed",
    ],
    keywords=[
        "amazon",
        "brand-intelligence",
        "ai",
        "llm",
        "claude",
        "langchain",
        "langgraph",
        "market-analysis",
        "e-commerce",
    ],
    license="MIT",
    zip_safe=False,
)
