#!/bin/bash
# Development Environment Setup Script for PyGeomodeling

set -e  # Exit on error

echo "=================================================="
echo "PyGeomodeling Development Environment Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}DONE: Python $python_version detected${NC}"
else
    echo -e "${RED}ERROR: Python 3.9+ required, found $python_version${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}[2/7] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}  Virtual environment already exists${NC}"
    read -p "  Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}DONE: Virtual environment recreated${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}DONE: Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[3/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}DONE: Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}[4/7] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}DONE: pip upgraded${NC}"

# Install package in development mode
echo -e "\n${YELLOW}[5/6] Installing package in development mode...${NC}"
pip install -e ".[dev,docs,all]"
echo -e "${GREEN}DONE: Package installed${NC}"

# Run initial tests
echo -e "\n${YELLOW}[6/6] Running initial tests...${NC}"
if pytest tests/ -v --tb=short -x; then
    echo -e "${GREEN}DONE: Tests passed${NC}"
else
    echo -e "${YELLOW}WARNING: Some tests failed (this may be expected)${NC}"
fi

# Summary
echo -e "\n=================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "=================================================="
echo -e "\nTo activate the environment in the future:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "\nUseful commands:"
echo -e "  ${YELLOW}pytest tests/${NC}              - Run tests"
echo -e "  ${YELLOW}black spe9_geomodeling/${NC}    - Format code"
echo -e "  ${YELLOW}flake8 spe9_geomodeling/${NC}   - Lint code"
echo -e "  ${YELLOW}mkdocs serve${NC}               - Build docs"
echo -e "  ${YELLOW}jupyter notebook${NC}           - Start Jupyter"
echo -e "\nNext steps:"
echo -e "  1. Read ${YELLOW}CONTRIBUTING.md${NC}"
echo -e "  2. Try ${YELLOW}examples/notebooks/${NC}"
echo -e "  3. Run ${YELLOW}python examples/advanced_workflow.py${NC}"
echo -e "\nHappy coding!\n"
