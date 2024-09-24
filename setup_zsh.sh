#!/bin/bash

# Update package lists
sudo apt update -y

# Install unzip and zsh
sudo apt install -y unzip zsh python3-pip python3.12-venv

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Optional: Set Zsh as the default shell
# chsh -s $(which zsh)

echo "Installation complete. Please log out and log back in to use Zsh with Oh My Zsh."