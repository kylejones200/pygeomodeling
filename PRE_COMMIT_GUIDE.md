# Pre-Commit Hooks Guide

## ✅ Setup Complete

Pre-commit hooks are now installed and will automatically check your code quality before every commit.

---

## 🎯 What Hooks Do

Every time you run `git commit`, the following checks run automatically:

### **Python Code Quality**

1. **isort** - Organizes imports (stdlib → third-party → local)
2. **autoflake** - Removes unused imports and variables
3. **black** - Formats code to consistent style (88 char lines)
4. **flake8** - Checks PEP8 compliance (with sensible ignores)
5. **pyupgrade** - Upgrades to modern Python 3.9+ syntax

### **File Quality**

6. **trailing-whitespace** - Removes trailing spaces
7. **end-of-file-fixer** - Ensures files end with newline
8. **check-yaml** - Validates YAML syntax
9. **check-json** - Validates JSON syntax
10. **check-toml** - Validates TOML syntax
11. **check-added-large-files** - Prevents committing large files (>1MB)
12. **check-merge-conflict** - Detects merge conflict markers
13. **detect-private-key** - Prevents committing API keys/secrets

### **Security**

14. **bandit** - Scans for security vulnerabilities

### **Documentation** (Optional)

15. **markdownlint** - Lints markdown files
16. **yamllint** - Lints YAML files

---

## 🚀 How It Works

### **Automatic on Commit**

```bash
git add my_file.py
git commit -m "feat: Add new feature"

# Pre-commit runs automatically:
# ✅ isort.....................Passed
# ✅ autoflake.................Passed
# ✅ black.....................Passed
# ✅ flake8....................Passed
# ... (all checks)
#
# If all pass: Commit succeeds ✅
# If any fail: Commit blocked ❌ (files fixed automatically when possible)
```

### **Manual Run**

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black --all-files
```

### **Skip Hooks (Use Sparingly)**

```bash
# Skip all hooks (NOT RECOMMENDED)
git commit --no-verify -m "message"

# Better: Fix the issues instead!
```

---

## 🛠️ Installation (Already Done!)

For new contributors or fresh clones:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Test it works
pre-commit run --all-files
```

---

## 📋 Hook Configuration

Located in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        language_version: python3.9
        args: [--line-length=88]
```

### **Ignored Issues (Intentional)**

flake8 ignores certain violations that are acceptable:

- **E203**: Whitespace before ':' (Black style)
- **W503**: Line break before binary operator (modern style)
- **E501**: Line too long (Black handles this)
- **F541**: f-string without placeholders (acceptable)
- **E402**: Module import not at top (delayed imports for optional deps)
- **E731**: Lambda expressions (acceptable for simple callbacks)
- **E712**: Comparison to True (JSON serialization)
- **E722**: Bare except (legacy compatibility)
- **F841**: Unused variable (matplotlib return values)

---

## 🔧 Troubleshooting

### **Hook Fails with "command not found"**

```bash
# Reinstall dependencies
pip install black isort autoflake flake8 pyupgrade bandit

# Update hooks
pre-commit autoupdate
```

### **Hook Takes Long Time First Run**

This is normal! Pre-commit:

1. Creates virtual environments for each hook
2. Downloads dependencies
3. Caches everything

**Subsequent runs are fast** (cached).

### **Want to Update Hooks?**

```bash
# Update to latest versions
pre-commit autoupdate

# Then commit the changes
git add .pre-commit-config.yaml
git commit -m "chore: Update pre-commit hooks"
```

### **Disable Specific Hook**

Edit `.pre-commit-config.yaml` and comment out or remove the hook:

```yaml
# repos:
#   - repo: https://github.com/igorshubovych/markdownlint-cli
#     rev: v0.38.0
#     hooks:
#       - id: markdownlint
#         args: [--fix]
```

---

## 📊 What Gets Fixed Automatically

### **Automatic Fixes:**

- ✅ Import organization (isort)
- ✅ Code formatting (black)
- ✅ Unused imports removed (autoflake)
- ✅ Trailing whitespace removed
- ✅ File endings fixed
- ✅ Markdown formatting (if enabled)

### **Requires Manual Fix:**

- ❌ PEP8 violations (flake8 reports them)
- ❌ Security issues (bandit flags them)
- ❌ Large files (need manual removal)
- ❌ Merge conflicts (need resolution)

---

## 🎯 Best Practices

### **1. Run Before Committing**

```bash
# Check your changes before commit
pre-commit run

# Or check everything
pre-commit run --all-files
```

### **2. Fix Issues, Don't Skip**

```bash
# Bad ❌
git commit --no-verify

# Good ✅
# Fix the reported issues, then commit normally
```

### **3. Keep Hooks Updated**

```bash
# Monthly or when issues occur
pre-commit autoupdate
```

### **4. Share with Team**

```bash
# Contributors just need to run:
pre-commit install

# Hooks are tracked in git (.pre-commit-config.yaml)
```

---

## 📈 Benefits

### **For You:**

- ✅ No more manual formatting
- ✅ Catch issues before CI/CD
- ✅ Consistent code style
- ✅ Learn best practices automatically

### **For Team:**

- ✅ No style debates (Black decides)
- ✅ Clean diffs (no formatting noise)
- ✅ Faster code reviews
- ✅ Higher code quality

### **For Project:**

- ✅ Professional appearance
- ✅ Easier to maintain
- ✅ Fewer bugs
- ✅ Better collaboration

---

## 🔍 Verification

Check if hooks are installed:

```bash
# List installed hooks
pre-commit install --install-hooks

# Show hook configuration
cat .git/hooks/pre-commit

# Test hooks work
echo "import os" > test.py
git add test.py
git commit -m "test"  # Should run hooks
git reset HEAD~1  # Undo test commit
rm test.py
```

---

## 📚 Additional Resources

### **Pre-commit Documentation:**

- Official Docs: <https://pre-commit.com/>
- Available Hooks: <https://pre-commit.com/hooks.html>
- Configuration: <https://pre-commit.com/#plugins>

### **Tool Documentation:**

- Black: <https://black.readthedocs.io/>
- isort: <https://pycqa.github.io/isort/>
- flake8: <https://flake8.pycqa.org/>
- autoflake: <https://github.com/PyCQA/autoflake>
- bandit: <https://bandit.readthedocs.io/>

---

## 🎉 Summary

**Pre-commit hooks are active!** They will:

1. ✅ **Automatically fix** formatting and imports
2. ✅ **Catch issues** before they reach CI/CD
3. ✅ **Enforce standards** consistently
4. ✅ **Save time** on code reviews
5. ✅ **Improve quality** incrementally

**Every commit now has built-in quality control!** 🚀

---

## 💡 Quick Commands

```bash
# Run hooks manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate

# Reinstall hooks (if issues)
pre-commit uninstall
pre-commit install

# Skip hooks (emergency only)
git commit --no-verify

# Check hook status
pre-commit --version
```

---

**Questions?** Check the official docs or run `pre-commit --help`

**Status**: ✅ **Active and Enforcing Quality**
