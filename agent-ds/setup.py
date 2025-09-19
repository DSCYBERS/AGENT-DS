#!/usr/bin/env python3
"""
Setup script for Agent DS - Autonomous Red-Team CLI AI Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agent-ds",
    version="1.0.0",
    author="Agent DS Development Team",
    author_email="agentds@government.classified",
    description="Autonomous Red-Team CLI AI Framework for Government-Authorized Operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://classified.gov/agent-ds",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Government",
        "License :: Classified",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: System :: Penetration Testing",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agent-ds=agent_ds:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "core": ["**/*.yaml", "**/*.json", "**/*.txt"],
    },
    zip_safe=False,
    keywords="penetration-testing security ai autonomous red-team government",
    project_urls={
        "Bug Reports": "https://classified.gov/agent-ds/issues",
        "Source": "https://classified.gov/agent-ds/source",
        "Documentation": "https://classified.gov/agent-ds/docs",
    },
)