#!/usr/bin/env python3
"""
Package preparation script for SPE9 Geomodeling Toolkit.

This script helps prepare your project for distribution.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.9/3.10
    import tomli as tomllib  # type: ignore[assignment]


def check_requirements():
    """Check if required packaging tools are installed."""
    required_packages = ["build", "twine"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error("Missing required packages: %s", ", ".join(missing))
        logger.info("Install with: pip install %s", " ".join(missing))
        return False

    logger.info("All packaging requirements satisfied")
    return True


def clean_build():
    """Clean previous build artifacts."""
    dirs_to_clean = ["build", "dist", "*.egg-info"]

    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                logger.info("Cleaned %s", path)


def build_package():
    """Build the package."""
    logger.info("Building package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "build"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("Package built successfully!")
            logger.info("Created files:")
            for file in Path("dist").glob("*"):
                logger.info("  - %s", file)
            return True
        else:
            logger.error("Build failed: %s", result.stderr)
            return False
    except Exception as e:
        logger.error("Build error: %s", e)
        return False


def create_archive():
    """Create a simple archive for sharing."""
    logger.info("Creating archive...")

    files_to_include = [
        "*.py",
        "pyproject.toml",
        "README.md",
        "requirements.txt",  # if it exists
    ]

    archive_files = []
    for pattern in files_to_include:
        archive_files.extend(Path(".").glob(pattern))

    # Create tar.gz archive
    import tarfile

    with tarfile.open("pygeomodeling-toolkit.tar.gz", "w:gz") as tar:
        for file in archive_files:
            if file.is_file():
                tar.add(file)
                logger.info("  + %s", file)

    logger.info("Archive created: pygeomodeling-toolkit.tar.gz")


def get_project_metadata() -> dict:
    """Read project metadata from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    return {
        "name": project.get("name", "pygeomodeling"),
        "version": project.get("version", "0.0.0"),
    }


def show_installation_instructions():
    """Show installation instructions for users."""
    metadata = get_project_metadata()
    dist_name = metadata.get("name", "pygeomodeling")
    version = metadata.get("version", "0.0.0")
    wheel_name = f"{dist_name.replace('-', '_')}-{version}-py3-none-any.whl"

    logger.info("INSTALLATION INSTRUCTIONS FOR USERS")

    logger.info("Option 1: Install from wheel (if built):")
    logger.info("pip install dist/%s", wheel_name)

    logger.info("Option 2: Install from source (core dependencies):")
    logger.info("pip install .")
    logger.info("pip install -e .  # development mode")

    logger.info("Option 3: Install with optional extras:")
    logger.info("pip install '.[dev]'        # development tooling")
    logger.info("pip install '.[advanced]'   # GPyTorch/Optuna workflow")
    logger.info("pip install '.[geospatial]' # Geo file I/O stack")
    logger.info("pip install '.[all]'        # Everything except GPU-only extras")

    logger.info("Option 4: From source archive:")
    logger.info("tar -xzf pygeomodeling-toolkit.tar.gz")
    logger.info("cd pygeomodeling-toolkit")
    logger.info("pip install -e .")


def main():
    """Run the main packaging workflow."""
    logger.info("SPE9 Geomodeling Toolkit - Package Preparation")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        logger.error("Error: pyproject.toml not found. Run this script from the project root.")
        return

    # Clean previous builds
    clean_build()

    # Check requirements
    if not check_requirements():
        logger.info("Note: Install missing packages and run again.")
        return

    # Build package
    if build_package():
        logger.info("Package ready for distribution!")

    # Create simple archive as backup
    create_archive()

    # Show instructions
    show_installation_instructions()

    logger.info("Your SPE9 geomodeling toolkit is ready to share!")
    logger.info("Files created in: %s", Path.cwd())


if __name__ == "__main__":
    main()
