#!/bin/bash
set -e

# Remotes
REMOTES=("origin" "bitbucket")

echo "🔄 Fetching latest changes from both remotes..."
for remote in "${REMOTES[@]}"; do
  git fetch "$remote"
done

# Track sync status
REPORT=""

# Loop through all local branches
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
  echo -e "\n➡️ Syncing branch: $branch"
  git checkout "$branch"

  # Pull from both remotes
  for remote in "${REMOTES[@]}"; do
    if git ls-remote --exit-code "$remote" "$branch" &>/dev/null; then
      echo "   📥 Pulling from $remote/$branch"
      git pull "$remote" "$branch" || true
    else
      echo "   ⚠️ $remote/$branch does not exist yet"
    fi
  done

  # Push to both remotes
  for remote in "${REMOTES[@]}"; do
    echo "   📤 Pushing to $remote/$branch"
    git push "$remote" "$branch" || true
  done

  # Compare with both remotes
  STATUS="✅ $branch is up to date"
  for remote in "${REMOTES[@]}"; do
    if git ls-remote --exit-code "$remote" "$branch" &>/dev/null; then
      LOCAL=$(git rev-parse "$branch")
      REMOTE=$(git rev-parse "$remote/$branch")
      if [ "$LOCAL" != "$REMOTE" ]; then
        STATUS="⚠️ $branch differs from $remote"
      fi
    else
      STATUS="⚠️ $branch missing on $remote"
    fi
  done

  REPORT="$REPORT\n$STATUS"
done

echo -e "\n📊 Final Sync Report:"
echo -e "$REPORT"

# Switch back to develop (or main)
git checkout develop || git checkout main
