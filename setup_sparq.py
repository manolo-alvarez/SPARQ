"""Setup script for SPARQ Agent."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README_SPARQ.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="sparq-agent",
    version="0.1.0",
    description="SPARQ: Similarity Prior with Adaptive Retrieval and Quick lookahead for AgentGym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Manolo Alvarez",
    author_email="manoloac@stanford.edu",
    url="https://github.com/manolo-alvarez/SPARQ",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "faiss": ["faiss-cpu>=1.7.0"],
        "faiss-gpu": ["faiss-gpu>=1.7.0"],
        "viz": ["matplotlib>=3.3.0", "pandas>=1.2.0"],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sparq-run-baseline=scripts.run_baseline:main",
            "sparq-run-ablations=scripts.run_ablations:main",
            "sparq-export-metrics=scripts.export_metrics:main",
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
        "Programming Language :: Python :: 3.11",
    ],
    keywords="reinforcement-learning agent planning retrieval memory",
)
