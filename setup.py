from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="creative-ai-evaluation-framework",
    version="0.1.0",
    author="Jason Derr",
    author_email="jason@example.com",
    description="The open-source standard for evaluating creative AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/derrtaderr/creative-ai-eval-framework",
    project_urls={
        "Bug Tracker": "https://github.com/derrtaderr/creative-ai-eval-framework/issues",
        "Documentation": "https://github.com/derrtaderr/creative-ai-eval-framework/docs",
        "Source Code": "https://github.com/derrtaderr/creative-ai-eval-framework",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "cloud": [
            "boto3>=1.24.0",
            "google-cloud-storage>=2.5.0",
            "azure-storage-blob>=12.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "creative-ai-eval=creative_ai_evaluation.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 