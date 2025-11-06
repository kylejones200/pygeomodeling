#!/bin/bash
# Build and optionally publish package to PyPI

set -e

echo "=================================================="
echo "PyGeomodeling Package Build & Publish"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ Not in a virtual environment${NC}"
    echo "Activate with: source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean previous builds
echo -e "\n${YELLOW}[1/5] Cleaning previous builds...${NC}"
rm -rf dist/ build/ *.egg-info
echo -e "${GREEN}✓ Cleaned${NC}"

# Install/upgrade build tools
echo -e "\n${YELLOW}[2/5] Installing build tools...${NC}"
pip install --upgrade build twine
echo -e "${GREEN}✓ Build tools ready${NC}"

# Build the package
echo -e "\n${YELLOW}[3/5] Building package...${NC}"
python -m build
echo -e "${GREEN}✓ Package built${NC}"

# Check the built package
echo -e "\n${YELLOW}[4/5] Checking package...${NC}"
twine check dist/*
echo -e "${GREEN}✓ Package validated${NC}"

# List built files
echo -e "\n${YELLOW}Built files:${NC}"
ls -lh dist/

# Ask about publishing
echo -e "\n${YELLOW}[5/5] Publish to PyPI?${NC}"
echo "Options:"
echo "  1) Test PyPI (recommended first)"
echo "  2) Production PyPI"
echo "  3) Skip publishing"
read -p "Choose (1/2/3): " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Publishing to Test PyPI...${NC}"
        echo "You'll need your Test PyPI credentials"
        twine upload --repository testpypi dist/*
        echo -e "${GREEN}✓ Published to Test PyPI${NC}"
        echo -e "\nTest installation with:"
        echo -e "  pip install --index-url https://test.pypi.org/simple/ pygeomodeling"
        ;;
    2)
        echo -e "\n${RED}⚠ Publishing to Production PyPI${NC}"
        read -p "Are you sure? This cannot be undone (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Publishing to PyPI...${NC}"
            twine upload dist/*
            echo -e "${GREEN}✓ Published to PyPI${NC}"
            echo -e "\nInstall with:"
            echo -e "  pip install pygeomodeling"
        else
            echo "Cancelled"
        fi
        ;;
    3)
        echo -e "${YELLOW}Skipping publish${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=================================================="
echo "Build Complete!"
echo "==================================================${NC}"
