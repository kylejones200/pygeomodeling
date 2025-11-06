# 🚀 Deploy v0.2.0 Now

## Quick Deploy (One Command)

```bash
./deploy.sh
```

This will handle everything automatically:
- Clean repo
- Build package
- Build docs
- Commit changes
- Create tag v0.2.0
- Push to GitHub
- Trigger PyPI auto-publish

## What Happens Next

1. **GitHub receives push** → Triggers workflows
2. **Test workflow runs** → Validates code quality
3. **Tag detected (v0.2.0)** → Triggers publish workflow
4. **Package builds** → Creates wheel and source dist
5. **PyPI publishes** → Package goes live
6. **Read the Docs** → Rebuilds documentation

## Timeline

- **Immediate**: Git push completes
- **~2-5 min**: GitHub Actions workflows start
- **~5-10 min**: Tests complete
- **~10-15 min**: Package published to PyPI
- **~15-20 min**: Read the Docs updates

## Verify Deployment

After running `./deploy.sh`, check:

```bash
# 1. GitHub Actions
open https://github.com/kylejones200/pygeomodeling/actions

# 2. PyPI (wait ~10 min)
open https://pypi.org/project/pygeomodeling/

# 3. Read the Docs (wait ~15 min)
open https://pygeomodeling.readthedocs.io/

# 4. Test install (wait ~10 min for PyPI)
pip install pygeomodeling==0.2.0
python -c "from spe9_geomodeling import SpatialKFold; print('✓ Works!')"
```

## If Something Goes Wrong

```bash
# Delete tag and retry
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0

# Fix issue, then redeploy
./deploy.sh
```

## Manual Deploy (If Needed)

```bash
# 1. Build
python -m build

# 2. Commit
git add -A
git commit -m "feat: release v0.2.0"

# 3. Tag
git tag -a v0.2.0 -m "Release v0.2.0"

# 4. Push
git push origin main
git push origin v0.2.0
```

---

**Ready?** Run: `./deploy.sh`
