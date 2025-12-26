"""
Setup script for Fish Health Monitoring System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fish-health-monitor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time fish health monitoring system using computer vision and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Fish-CNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "filterpy>=1.4.5",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "websockets>=11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fish-monitor=examples.live_camera_monitoring:main",
        ],
    },
)
