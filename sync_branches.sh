#!/bin/bash
set -e
GITHUB="origin"
BITBUCKET="bitbucket"
FEATURE_BRANCH="feature/ollama-langchain-lab"
DEVELOP_BRANCH="develop"
MAIN_BRANCH="main"

echo "🔄 Fetching all remotes..."
git fetch $GITHUB
git fetch $BITBUCKET

# -----------------------------------------------------------------------------
# Step 1: Ensure feature branch is updated
# -----------------------------------------------------------------------------
echo "🌿 Checking out $FEATURE_BRANCH..."
git checkout $FEATURE_BRANCH
git pull --rebase $GITHUB $FEATURE_BRANCH || true
git pull --rebase $BITBUCKET $FEATURE_BRANCH || true

# -----------------------------------------------------------------------------
# Step 2: Merge feature → main
# -----------------------------------------------------------------------------
echo "📦 Updating $MAIN_BRANCH with $FEATURE_BRANCH..."
git checkout $MAIN_BRANCH
git pull --rebase $GITHUB $MAIN_BRANCH || true
git pull --rebase $BITBUCKET $MAIN_BRANCH || true
git merge --no-ff $FEATURE_BRANCH -m "Merge $FEATURE_BRANCH into $MAIN_BRANCH"

git push $GITHUB $MAIN_BRANCH
git push $BITBUCKET $MAIN_BRANCH

# -----------------------------------------------------------------------------
# Step 3: Rebase develop → main
# -----------------------------------------------------------------------------
echo "🔧 Rebasing $DEVELOP_BRANCH onto $MAIN_BRANCH..."
git checkout $DEVELOP_BRANCH
git pull --rebase $GITHUB $DEVELOP_BRANCH || true
git pull --rebase $BITBUCKET $DEVELOP_BRANCH || true
git rebase $MAIN_BRANCH

git push $GITHUB $DEVELOP_BRANCH --force
git push $BITBUCKET $DEVELOP_BRANCH --force

# -----------------------------------------------------------------------------
# Step 4: Rebase feature → main
# -----------------------------------------------------------------------------
echo "🌟 Rebasing $FEATURE_BRANCH onto $MAIN_BRANCH..."
git checkout $FEATURE_BRANCH
git rebase $MAIN_BRANCH

git push $GITHUB $FEATURE_BRANCH --force
git push $BITBUCKET $FEATURE_BRANCH --force

# -----------------------------------------------------------------------------
# Step 5: Report status
# -----------------------------------------------------------------------------
echo ""
echo "✅ Sync complete! Branch status:"
git fetch --all --quiet
git branch -vv
git log --oneline --graph --decorate --all --max-count=15
