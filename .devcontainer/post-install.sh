#!/bin/zsh -l
# Add the current directory to Git's safe directory list
git config --global --add safe.directory /workspaces/OpenEnergyID

# Install Python tools
pipx install ruff poetry pre-commit

# Install project dependencies using Poetry
poetry install

# Install pre-commit hooks
pre-commit install

# Install Node.js and Pure prompt
# nvm install node
# npm install --global pure-prompt

# # Configure Zsh to use Pure prompt
# "autoload -U promptinit; promptinit; prompt pure"
# "echo "autoload -U promptinit; promptinit; prompt pure" >> ~/.zshrc"

# # Uncomment the following line to install zsh-syntax-highlighting
# # sudo apt-get install zsh-syntax-highlighting
# # echo "source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc
