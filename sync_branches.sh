#!/bin/bash

# Branches you want to keep in sync
BRANCHES=("develop" "main")

# Fetch all latest changes from both remotes
echo "📥 Fetching latest changes from origin (GitHub) and bitbucket..."
git fetch --all --prune

# Sync core branches
for BR in "${BRANCHES[@]}"; do
    echo -e "\n🔄 Syncing branch: $BR"
    git checkout $BR || { echo "❌ Failed to checkout $BR"; exit 1; }

    # Pull latest from both remotes (if they exist)
    git pull origin $BR || echo "⚠️ No $BR on origin"
    git pull bitbucket $BR || echo "⚠️ No $BR on bitbucket"

    # Push branch to both remotes
    git push origin $BR || echo "⚠️ Could not push to origin"
    git push bitbucket $BR || echo "⚠️ Could not push to bitbucket"
done

# Sync all feature branches
echo -e "\n🔄 Syncing all feature/* branches..."
for BR in $(git branch --list "feature/*" | sed 's/* //'); do
    echo -e "\n➡️  Processing $BR"
    git checkout $BR || continue

    git pull origin $BR || echo "⚠️ No $BR on origin"
    git pull bitbucket $BR || echo "⚠️ No $BR on bitbucket"

    git push origin $BR || echo "⚠️ Could not push $BR to origin"
    git push bitbucket $BR || echo "⚠️ Could not push $BR to bitbucket"
done

# Final status report
echo -e "\n📊 Final Branch Status:"
git branch -vv
