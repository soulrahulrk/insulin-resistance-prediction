# GitHub Upload Script for Windows PowerShell
# Save as: upload_to_github.ps1 and run: .\upload_to_github.ps1

Write-Host "🚀 Insulin Resistance Prediction - GitHub Upload (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will help you push your project to GitHub." -ForegroundColor Yellow
Write-Host ""

$GitHubUser = Read-Host "Enter your GitHub username"
$RepoName = Read-Host "Enter repository name (press Enter for 'insulin-resistance-prediction')"
if ([string]::IsNullOrWhiteSpace($RepoName)) {
    $RepoName = "insulin-resistance-prediction"
}

$RepoUrl = "https://github.com/$GitHubUser/$RepoName.git"

Write-Host ""
Write-Host "📋 Configuration:" -ForegroundColor Cyan
Write-Host "  GitHub User: $GitHubUser"
Write-Host "  Repo Name: $RepoName"
Write-Host "  Full URL: $RepoUrl"
Write-Host ""
Write-Host "✅ Steps to complete BEFORE running this script:" -ForegroundColor Yellow
Write-Host "   1. Go to https://github.com/new"
Write-Host "   2. Create a new repository with the same name"
Write-Host "   3. Do NOT add README, .gitignore, or License (we have them)"
Write-Host "   4. Click 'Create repository'"
Write-Host ""

$Continue = Read-Host "Have you created the repo on GitHub? (y/n)"
if ($Continue -ne "y") {
    Write-Host "❌ Please create the repository on GitHub first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔧 Executing git commands..." -ForegroundColor Cyan
Write-Host ""

# Add remote
Write-Host "→ Adding remote..."
git remote add origin "$RepoUrl"

# Rename branch to main (optional)
Write-Host "→ Renaming branch to main..."
git branch -M main

# Verify remote
Write-Host "→ Verifying configuration..."
Write-Host ""
git remote -v

Write-Host ""
Write-Host "📤 Pushing to GitHub (you may be prompted to authenticate)..." -ForegroundColor Yellow
Write-Host "   This may take a minute..." -ForegroundColor Gray
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ SUCCESS! Your project is now on GitHub!" -ForegroundColor Green
    Write-Host "   Visit: $RepoUrl" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 Next actions:" -ForegroundColor Cyan
    Write-Host "   • Update repo settings on GitHub"
    Write-Host "   • Add collaborators if needed"
    Write-Host "   • Enable branch protection on 'main' branch"
    Write-Host "   • Monitor CI/CD in the Actions tab"
} else {
    Write-Host ""
    Write-Host "❌ Error during push. Check the output above." -ForegroundColor Red
    Write-Host "   Common issues:"
    Write-Host "   • Authentication failed: Check GitHub credentials"
    Write-Host "   • Remote already exists: Run 'git remote remove origin' first"
}
