"""Setup script for oscar"""
from setuptools import setup, find_packages

setup(
    name='oscar',
    version='0.0.1',
    description='GPU-accelerated simulation and reinforcement learning toolkit for data-driven robot controllers',
    author='NVIDIA CORPORATION',
    author_email=['jdwong@stanford.edu', 'vmakoviychuk@nvidia.com', 'aanandkumar@nvidia.com', 'yukez@nvidia.com'],
    url='',
    license='Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.',
    packages=[package for package in find_packages() if package.startswith("oscar")],
    python_requires='>=3.6,<3.8',
    install_requires=[
        "torch>=1.4.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "scipy>=1.5.0",
        "pyyaml>=5.3.1",
        "gym>=0.17.1",
        "tensorboard>=2.2.1",
        "pillow",
        "imageio",
        "ninja",
        "isaacgym",
        "trimesh",
        "matplotlib",
    ],
)
