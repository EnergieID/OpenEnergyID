# Development Environment Setup

This guide outlines the setup process for our development environment, focusing on packaging and dependency management.

## Info on dev setup

### Commit practices (pre-commit)

we use [pre-commit ](https://pre-commit.com/)to run RUFF.

### uv for Packaging and Dependency Management

We use [uv](https://github.com/astral-sh/uv) as our primary tool for packaging and managing dependencies. uv is an extremely fast Python package installer and resolver, written in Rust.

For the most detailed and up-to-date information, please refer to the [official uv documentation](https://astral.sh/guide/uv).

## Setup with uv

1. First, [install uv](https://astral.sh/guide/uv#installation) following the instructions on the official website.
2. once uv is installed, you can create a virtual environment and install the dependencies with the following commands:

```shell
uv venv
uv sync
```

## Remarks for devcontainer

You can also work inside a docker container(devcontainer).

```bash
whoami
sudo chown -R $USER:$USER /workspaces/ -R
```

### some info about devcontainers:

- [our python/container version](https://github.com/devcontainers/images/blob/main/src/python/history/1.1.9.md#variant-311-bookworm)

- notes on [devcontainer git authentication](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials). dont forget to ssh-add if you use keys.

# Useful commands for dev work

## clean up

[vulture](https://github.com/jendrikseipp/vulture) for finding dead python 🐍
`uv run vulture . --exclude venv`

[deptry](https://github.com/fpgmaas/deptry) for checking dependencies
`uv run deptry . --ignore-notebooks`
