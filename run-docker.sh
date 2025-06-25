#!/bin/bash

# Detect hardware and run appropriate Docker configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}BON Jailbreak Docker Runner${NC}"
echo "================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env file with your HF_TOKEN before running.${NC}"
    exit 1
fi

# Detect hardware
detect_hardware() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA GPU detected${NC}"
        return 1
    fi
    
    # Check for Apple Silicon
    if [[ $(uname -s) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
        echo -e "${GREEN}Apple Silicon detected${NC}"
        return 2
    fi
    
    # Check for other ARM64
    if [[ $(uname -m) == "arm64" ]] || [[ $(uname -m) == "aarch64" ]]; then
        echo -e "${GREEN}ARM64 architecture detected${NC}"
        return 3
    fi
    
    # Default to CPU
    echo -e "${YELLOW}No GPU detected, using CPU${NC}"
    return 0
}

# Parse command line arguments
PROFILE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Auto-detect if no profile specified
if [ -z "$PROFILE" ]; then
    detect_hardware
    HARDWARE=$?
    
    case $HARDWARE in
        1)
            PROFILE="gpu"
            export DOCKER_RUNTIME=nvidia
            ;;
        2)
            PROFILE="arm64"
            ;;
        3)
            PROFILE="arm64"
            ;;
        0)
            PROFILE="cpu"
            ;;
    esac
fi

echo -e "${GREEN}Using profile: $PROFILE${NC}"

# Build and run
echo -e "${GREEN}Building Docker image...${NC}"
docker-compose --profile $PROFILE build

echo -e "${GREEN}Starting experiment...${NC}"
if [ -n "$EXTRA_ARGS" ]; then
    docker-compose --profile $PROFILE run --rm bon-$PROFILE $EXTRA_ARGS
else
    docker-compose --profile $PROFILE up
fi

echo -e "${GREEN}Experiment complete!${NC}"
echo "Results are in the ./output directory"