# Uploading to GitHub: Step-by-Step Guide

## Summary of What Was Done ✅

1. **Consolidated documentation:** Merged all technical docs into a single, comprehensive `README.md`
2. **Cleaned up redundant files:** Removed 9 redundant markdown files (kept only `RUNBOOK.md` and `PRIVACY_CHECKLIST.md`)
3. **Added production files:**
   - `.gitignore` – excludes .venv, data/, logs/, metrics/, models/*.pkl, etc.
   - `LICENSE` – MIT license
   - `CONTRIBUTING.md` – contribution guidelines
4. **Initialized Git locally:** Initial commit with 58 files

## Current Git Status

```
Repository: c:\Users\rahul\Documents\code\projects\ir prediction
Branch: master (root-commit ready to push)
Commits: 1 (Initial commit)
Files tracked: 58
```

---

## Upload to GitHub: 4 Easy Steps

### Step 1: Create Repository on GitHub

1. Go to https://github.com and log in to your account
2. Click the **+** (top-right corner) → **New repository**
3. Fill in:
   - **Repository name:** `insulin-resistance-prediction` (or your preference)
   - **Description:** "Clinical ML system for insulin resistance prediction using ensemble stacking, drift detection, and SHAP explanations"
   - **Public/Private:** Choose your preference
   - **Do NOT add README, .gitignore, or License** (you already have them locally)
4. Click **Create repository**

GitHub will show you a page with commands. **Copy these URL steps** (replace `your-username` and `repo-name`):

```
https://github.com/your-username/insulin-resistance-prediction
```

---

### Step 2: Connect Local Repo to GitHub

Open **PowerShell** and run these commands from your project directory:

```powershell
cd "C:\Users\rahul\Documents\code\projects\ir prediction"

# Add remote (replace with your GitHub URL from Step 1)
git remote add origin https://github.com/your-username/insulin-resistance-prediction.git

# Rename branch to main (optional but recommended)
git branch -M main

# Verify remote is set correctly
git remote -v
```

Expected output:
```
origin  https://github.com/your-username/insulin-resistance-prediction.git (fetch)
origin  https://github.com/your-username/insulin-resistance-prediction.git (push)
```

---

### Step 3: Push to GitHub

```powershell
# Push the initial commit and set upstream
git push -u origin main
```

First time you run this, GitHub will prompt you to authenticate:
- **Option A (Browser login):** A browser window opens; authorize and return to terminal
- **Option B (Personal Access Token):** If using a token, paste it when prompted

---

### Step 4: Verify on GitHub

1. Refresh your GitHub repo page: `https://github.com/your-username/insulin-resistance-prediction`
2. Confirm you see:
   - All 58 files (src/, tests/, scripts/, docs/, etc.)
   - README.md displayed as the main page
   - License and Contributing files visible

---

## Ongoing GitHub Workflow

### After Making Changes Locally

```powershell
cd "C:\Users\rahul\Documents\code\projects\ir prediction"

# Check what changed
git status

# Stage changes
git add .

# Commit with a message
git commit -m "Feature: add real-time monitoring dashboard"

# Push to GitHub
git push
```

### Common Commands

```powershell
# View commit history
git log --oneline

# Create a new branch for features
git checkout -b feature/your-feature-name

# Switch back to main
git checkout main

# Merge a feature branch
git merge feature/your-feature-name

# Pull latest changes from GitHub
git pull
```

---

## Optional: Setup Branch Protection (Recommended)

1. Go to your GitHub repo → **Settings** → **Branches**
2. Under "Branch protection rules," click **Add rule**
3. Enter branch name: `main`
4. Enable:
   - ✅ "Require pull request reviews before merging"
   - ✅ "Require status checks to pass" (links to CI/CD)
5. Click **Create**

This ensures code review before merging to production.

---

## Optional: Enable GitHub Pages (for Documentation)

If you want to host docs at `username.github.io/insulin-resistance-prediction`:

1. Go to repo → **Settings** → **Pages**
2. Under "Source," select **Deploy from a branch**
3. Select **main** branch and **/root** folder
4. Click **Save**

Your docs will be live in ~1 minute at the GitHub Pages URL.

---

## Checklist Before Pushing

- ✅ Git initialized locally
- ✅ Initial commit created (58 files)
- ✅ Repository created on GitHub
- ✅ Remote URL added locally
- ✅ Branch renamed to `main` (optional)
- ✅ `git push -u origin main` executed
- ✅ GitHub repo verified with all files

---

## Troubleshooting

**"fatal: could not read Username for 'https://github.com'"**
- Solution: GitHub redirects to browser login or token prompt. Complete authentication in the browser or enter your Personal Access Token.

**"rejected ... (fetch first)"**
- Solution: Run `git pull origin main` first, then `git push`

**"error: remote origin already exists"**
- Solution: Run `git remote remove origin` then add again with correct URL

**"Branch 'main' does not exist yet"**
- Solution: You're still on `master`. Run `git branch -M main` to rename.

---

## Next Steps After Upload

1. **Protect main branch:** Add branch protection rules (see above)
2. **Add collaborators:** Settings → Collaborators (if team project)
3. **Setup CI/CD:** Already configured in `.github/workflows/ci.yml`
4. **Create releases:** Tag versions for releases: `git tag -a v1.0.0 -m "Release 1.0"`
5. **Monitor:** Check GitHub Actions for CI/CD status on every push

---

**You're all set!** Your Insulin Resistance Prediction System is now professionally organized and ready for GitHub. 🚀

For questions or issues with Git/GitHub, refer to https://docs.github.com
