#!/usr/bin/env python3
"""
Setup script for Brain 4 Docling module
"""

from setuptools import setup, find_packages

setup(
    name="brain4_docling",
    version="1.0.0",
    description="Brain 4 Docling - Document Processing and Analysis Module",
    author="Four-Brain AI System",
    author_email="brain4@fourbrain.ai",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "redis>=5.0.0",
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=2.0.0",
        "docling>=1.0.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "prometheus-client>=0.20.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "pynvml>=11.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "brain4-docling=brain4_docling.main:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, document-processing, docling, brain4, four-brain-architecture",
    project_urls={
        "Documentation": "https://github.com/fourbrain/brain4-docling/docs",
        "Source": "https://github.com/fourbrain/brain4-docling",
        "Tracker": "https://github.com/fourbrain/brain4-docling/issues",
    },
)
