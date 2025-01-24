#!/bin/zsh -l
# Add the current directory to Git's safe directory list
git config --global --add safe.directory /workspaces/OpenEnergyID

# Install Python tools
pipx install ruff poetry pre-commit

# Install project dependencies using Poetry
poetry install

# Install pre-commit hooks
pre-commit install

# Install ZSH plugins
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Update .zshrc to enable plugins
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Source the updated configuration
source ~/.zshrc
