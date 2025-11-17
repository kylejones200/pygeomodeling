#!/usr/bin/env python3
"""
Package preparation script for SPE9 Geomodeling Toolkit.

This script helps prepare your project for distribution.
"""

import shutil
import subprocess
import sys
from pathlib import Path

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
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("✅ All packaging requirements satisfied")
    return True


def clean_build():
    """Clean previous build artifacts."""
    dirs_to_clean = ["build", "dist", "*.egg-info"]

    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"🧹 Cleaned {path}")


def build_package():
    """Build the package."""
    print("🔨 Building package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "build"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Package built successfully!")
            print("📦 Created files:")
            for file in Path("dist").glob("*"):
                print(f"  - {file}")
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False


def create_archive():
    """Create a simple archive for sharing."""
    print("📦 Creating archive...")

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

    with tarfile.open("spe9-geomodeling-toolkit.tar.gz", "w:gz") as tar:
        for file in archive_files:
            if file.is_file():
                tar.add(file)
                print(f"  + {file}")

    print("✅ Archive created: spe9-geomodeling-toolkit.tar.gz")


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

    print("\n" + "=" * 60)
    print("📋 INSTALLATION INSTRUCTIONS FOR USERS")
    print("=" * 60)

    print("\n🎯 Option 1: Install from wheel (if built):")
    print(f"pip install dist/{wheel_name}")

    print("\n🎯 Option 2: Install from source (core dependencies):")
    print("pip install .")
    print("pip install -e .  # development mode")

    print("\n🎯 Option 3: Install with optional extras:")
    print("pip install '.[dev]'        # development tooling")
    print("pip install '.[advanced]'   # GPyTorch/Optuna workflow")
    print("pip install '.[geospatial]' # Geo file I/O stack")
    print("pip install '.[all]'        # Everything except GPU-only extras")

    print("\n🎯 Option 4: From source archive:")
    print("tar -xzf spe9-geomodeling-toolkit.tar.gz")
    print("cd spe9-geomodeling-toolkit")
    print("pip install -e .")


def main():
    """Run the main packaging workflow."""
    print("🚀 SPE9 Geomodeling Toolkit - Package Preparation")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run this script from the project root.")
        return

    # Clean previous builds
    clean_build()

    # Check requirements
    if not check_requirements():
        print("\n💡 Install missing packages and run again.")
        return

    # Build package
    if build_package():
        print("\n✅ Package ready for distribution!")

    # Create simple archive as backup
    create_archive()

    # Show instructions
    show_installation_instructions()

    print(f"\n🎉 Your SPE9 geomodeling toolkit is ready to share!")
    print(f"📁 Files created in: {Path.cwd()}")


if __name__ == "__main__":
    main()
