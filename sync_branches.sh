#!/bin/bash
set -e  # Exit if any command fails

# Define remotes
GITHUB="origin"
BITBUCKET="bitbucket"

# Branches to sync
BRANCHES=("main" "develop" "feature/ollama-langchain-lab")

echo "🔄 Fetching latest changes from both remotes..."
git fetch $GITHUB
git fetch $BITBUCKET

echo ""
echo "=================================="
echo "📊 Branch Sync Status Report"
echo "=================================="

# Report current branch sync status
for branch in "${BRANCHES[@]}"; do
  echo "➡️  $branch:"
  git rev-list --left-right --count $GITHUB/$branch...$BITBUCKET/$branch || echo "   ⚠️  Branch missing in one of the remotes"
done

echo ""
echo "=================================="
echo "🚀 Starting Sync Process"
echo "=================================="

for branch in "${BRANCHES[@]}"; do
  echo ""
  echo "=============================="
  echo "📌 Syncing branch: $branch"
  echo "=============================="

  # Checkout branch
  git checkout $branch

  # Step 1: Rebase with GitHub remote branch
  echo "⬆️ Rebasing with $GITHUB/$branch..."
  git pull --rebase $GITHUB $branch || {
    echo "❌ Rebase failed with $GITHUB/$branch. Resolve conflicts manually."
    exit 1
  }

  # Step 2: Rebase on top of latest main (skip if branch is main)
  if [ "$branch" != "main" ]; then
    echo "⬆️ Rebasing $branch on top of $GITHUB/main..."
    git rebase $GITHUB/main || {
      echo "❌ Rebase on main failed. Resolve conflicts manually."
      exit 1
    }
  fi

  # Step 3: Push updated branch to both remotes
  echo "🚀 Pushing $branch to GitHub and Bitbucket..."
  git push $GITHUB $branch
  git push $BITBUCKET $branch
done

echo ""
echo "=================================="
echo "✅ Final Branch Status Report"
echo "=================================="

for branch in "${BRANCHES[@]}"; do
  echo "➡️  $branch:"
  git rev-list --left-right --count $GITHUB/$branch...$BITBUCKET/$branch || echo "   ⚠️  Branch missing in one of the remotes"
done

echo ""
echo "✅ All branches synced successfully!"
