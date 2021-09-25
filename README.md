## OSCAR
This repository contains the codebase used in [OSCAR: Data-Driven Operational Space Control for Adaptive and Robust Robot Manipulation](arxiv_link_TODO).

More generally, this codebase is a modular framework built upon [IsaacGym](https://developer.nvidia.com/isaac-gym), and intended to support future robotics research leveraging large-scale training.

Of note, this repo contains:
- High-quality controller implementations of OSC, IK, and Joint-Based controllers that have been fully parallelized for PyTorch
- Complex Robot Manipulation tasks for benchmarking learning algorithms
- Modular structure enabling rapid prototyping of additional robots, controllers, and environments

## Requirements
- Linux machine
- Conda
- NVIDIA GPU + CUDA

## Getting Started
First, clone this repo:

```bash
git clone https://github.com/NVlabs/oscar.git
```

Next, create a new conda environment to be used for this repo and activate the repo:
```bash
cd oscar
bash create_conda_env_oscar.sh
conda activate oscar
```

This will create a new conda environment named `oscar` and additional install some dependencies. Next, we need IsaacGym. This repo itself does not contain IsaacGym, but is compatible with any version > preview 3.0.

Install and build IsaacGym [HERE](https://developer.nvidia.com/isaac-gym).

Once installed, navigate to the python directory and install the package to this conda environment:

```bash
cd <ISAACGYM_REPO_PATH>/python
pip install -e .
```

Now with IsaacGym installed, we can finally install this repo as a package:

```bash
cd <OSCAR_REPO_PATH>
pip install -e .
```

That's it!

## Prerequisites

- Supported platforms are Linux (Ubuntu 16.04+) and Windows 10 (version 1809 is required).

- Install Git and [Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation).

- If you wish to develop using Isaac Gym, you should [fork this repository](https://gitlab-master.nvidia.com/carbon-gym/carbgym/forks/new).
  * Go to your newly created fork in GitLab, select
      * go to "Settings->Repository->Mirroring repositories"
          * set "Git repository URL" to https://gitlab-master.nvidia.com/carbon-gym/carbgym.git
          * select "Pull" under "Mirror direction".
          * clear out the text under "Password".
          * check the "Overwrite diverged branches" checkbox.
      * go to "Settings->General->Visibility, project features, permissions"
          * ensure "Project Visibility" is set to "Public".
  * Clone your fork to a local hard drive.  On Windows, make sure to use a NTFS drive.

- If you just want to use Isaac Gym as-is, you can clone this repositiory directly to your hard drive (on Windows, make sure to use a NTFS drive).

- On Ubuntu, you should run `./setup.sh` to install some prerequisite software.  This only needs to be done once. 


## Building

Execute `./build.sh` on Linux or `build.bat` on Windows.

For building in Visual Studio you need VS 2017 with [SDK 10.17763+](https://go.microsoft.com/fwlink/?LinkID=2023014).

The build output will be found in the generated `_build` folder and the make/solution 
files will be found in the generated `_compiler` directory. 

## Python

See the [Python README](/python/README.md) for information on running the Python examples.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md).
