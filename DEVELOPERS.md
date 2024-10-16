# Development Environment Setup

This guide outlines the setup process for our development environment, focusing on packaging and dependency management.

## Info on dev setup

### Commit practices (pre-commit)

we use [pre-commit ](https://pre-commit.com/)to run RUFF.

### Poetry for Packaging and Dependency Management

We use [Poetry](https://python-poetry.org/docs/main/#installation) as our primary tool for packaging and managing dependencies. Poetry provides a simple yet powerful way to manage project dependencies and publish packages.

For the most detailed and up-to-date information, please refer to the [official Poetry 📚](https://python-poetry.org/docs/main/#installation).

### Installing Python CLI Applications Globally with pipx

To ensure that Python CLI applications are installed globally on your system while being isolated in their own virtual environments, we utilize `pipx`.

## Steps to Install pipx and Poetry

1. First, [install pipx](https://pipx.pypa.io/stable/installation/) following the instructions on the official website.
2. Once pipx is installed, you can easily install Poetry by running the following command in your terminal:
   `pipx install poetry`
3. (_optional_) [install](vscode:extension/zeshuaro.vscode-python-poetry) vscode extension for poetry

### Setup with Poetry

```shell
poetry install
```

### publishing with poetry

https://python-poetry.org/docs/repositories/

## Remarks for devcontainer

You can also work inside a docker container(devcontainer).

```bash
whoami
sudo chown -R $USER:$USER /workspaces/ -R
```

### some info about devcontainers:

- [poetry in a devcontainer](https://marioscalas.medium.com/using-python-and-poetry-inside-a-dev-container-33c80bc5a22c)

- [our python/container version](https://github.com/devcontainers/images/blob/main/src/python/history/1.1.9.md#variant-311-bookworm)

- notes on [devcontainer git authentication](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials). dont forget to ssh-add if you use keys.

# Useful commands for dev work

## clean up

[vulture](https://github.com/jendrikseipp/vulture) for finding dead python 🐍
`pipx run vulture . --exclude venv`

[deptry](https://github.com/fpgmaas/deptry) for checking dependencies
`pipx run deptry . --ignore-notebooks`
