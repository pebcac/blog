#!/bin/bash
set -e

echo "=== Blog rebuild started at $(date) ==="

cd /site

# Pull latest changes
echo "Pulling latest changes..."
git fetch origin
git reset --hard origin/main

# Update submodules (theme)
echo "Updating theme submodule..."
git submodule update --init --recursive

# Build with Hugo
echo "Building site with Hugo..."
docker run --rm -v "$(pwd)":/src klakegg/hugo:ext-alpine --minify

echo "=== Blog rebuilt successfully at $(date) ==="
