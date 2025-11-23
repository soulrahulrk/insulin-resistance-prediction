#!/bin/bash
# GitHub Upload Script (Optional Quick Start)
# Save as: upload_to_github.sh and run: bash upload_to_github.sh

echo "🚀 Insulin Resistance Prediction - GitHub Upload"
echo "=================================================="
echo ""
echo "This script will help you push your project to GitHub."
echo ""
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter repository name (default: insulin-resistance-prediction): " REPO_NAME
REPO_NAME=${REPO_NAME:-insulin-resistance-prediction}

REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo ""
echo "📋 Configuration:"
echo "  GitHub User: $GITHUB_USER"
echo "  Repo Name: $REPO_NAME"
echo "  Full URL: $REPO_URL"
echo ""
echo "✅ Steps to complete BEFORE running this script:"
echo "   1. Go to https://github.com/new"
echo "   2. Create a new repository with the same name"
echo "   3. Do NOT add README, .gitignore, or License (we have them)"
echo "   4. Click 'Create repository'"
echo ""
read -p "Have you created the repo on GitHub? (y/n): " CREATED

if [ "$CREATED" != "y" ]; then
    echo "❌ Please create the repository on GitHub first."
    exit 1
fi

echo ""
echo "🔧 Executing git commands..."
echo ""

# Add remote
echo "→ Adding remote..."
git remote add origin "$REPO_URL"

# Rename branch to main (optional)
echo "→ Renaming branch to main..."
git branch -M main

# Verify remote
echo "→ Verifying configuration..."
git remote -v

echo ""
echo "📤 Pushing to GitHub (you may be prompted to authenticate)..."
git push -u origin main

echo ""
echo "✅ SUCCESS! Your project is now on GitHub!"
echo "   Visit: $REPO_URL"
echo ""
echo "📝 Next actions:"
echo "   - Update repo settings on GitHub"
echo "   - Add collaborators if needed"
echo "   - Enable branch protection on 'main'"
echo "   - Monitor CI/CD in Actions tab"
