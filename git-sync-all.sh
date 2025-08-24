#!/bin/bash
# git-sync-all.sh
# Sync all local branches with both GitHub (origin) and Bitbucket (bitbucket)

set -e  # stop on error

echo "🔄 Fetching from both remotes..."
git fetch origin --prune
git fetch bitbucket --prune

# Loop through each local branch
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
    echo "----------------------------"
    echo "🌿 Checking branch: $branch"

    # Checkout branch
    git checkout $branch

    # Merge latest from origin
    if git ls-remote --heads origin $branch | grep -q $branch; then
        echo "⬇️  Pulling latest from origin/$branch"
        git pull origin $branch || true
    else
        echo "⚠️  Branch $branch not found on origin"
    fi

    # Merge latest from bitbucket
    if git ls-remote --heads bitbucket $branch | grep -q $branch; then
        echo "⬇️  Pulling latest from bitbucket/$branch"
        git pull bitbucket $branch || true
    else
        echo "⚠️  Branch $branch not found on bitbucket"
    fi

    # Push branch to both remotes
    echo "⬆️  Pushing $branch → origin"
    git push origin $branch

    echo "⬆️  Pushing $branch → bitbucket"
    git push bitbucket $branch
done

# Return to develop (default branch)
git checkout develop
echo "✅ All branches synced with GitHub & Bitbucket!"
