#!/bin/bash

# JARVIS 10/10 - Seamless Setup Redirector
# This redirects to the new ultimate setup

echo "🚀 Redirecting to JARVIS 10/10 Ultimate Setup..."
echo

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Run the new setup
./setup_10_seamless.sh "$@"