#!/bin/bash
# sync_branches.sh - Check + Auto-fix branch sync across origin, bitbucket, and main

set -e

BRANCHES=("main" "develop" "feature/ollama-langchain-lab")

echo "🔄 Fetching latest changes from both remotes..."
git fetch origin
git fetch bitbucket

echo
echo "📊 Checking branch sync status..."
echo "----------------------------------------"

for branch in "${BRANCHES[@]}"; do
  echo "🔍 Checking $branch..."
  STATUS=""

  LOCAL=$(git rev-parse $branch 2>/dev/null || echo "none")
  ORIGIN=$(git rev-parse origin/$branch 2>/dev/null || echo "none")
  BITBUCKET=$(git rev-parse bitbucket/$branch 2>/dev/null || echo "none")

  # Compare local vs origin
  if [ "$LOCAL" != "$ORIGIN" ]; then
    STATUS+=" ⚠️ differs from origin"
  else
    STATUS+=" ✅ matches origin"
  fi

  # Compare local vs bitbucket
  if [ "$LOCAL" != "$BITBUCKET" ]; then
    STATUS+=" ⚠️ differs from bitbucket"
  else
    STATUS+=" ✅ matches bitbucket"
  fi

  # Ahead/Behind vs main
  if [ "$branch" != "main" ] && git rev-parse --verify main &>/dev/null; then
    BEHIND=$(git rev-list --count $branch..main || echo 0)
    AHEAD=$(git rev-list --count main..$branch || echo 0)

    if [ "$AHEAD" -gt 0 ] || [ "$BEHIND" -gt 0 ]; then
      STATUS+=" | 🔄 ahead $AHEAD / behind $BEHIND vs main"
    else
      STATUS+=" | ✅ in sync with main"
    fi
  fi

  echo "$branch -> $STATUS"
  echo "----------------------------------------"

  # Ask for fixing if differences found
  if [[ "$STATUS" == *"⚠️"* || "$STATUS" == *"ahead"* || "$STATUS" == *"behind"* ]]; then
    read -p "⚡ Do you want to sync $branch now? (y/n): " answer
    if [[ "$answer" == "y" ]]; then
      echo "➡️ Syncing $branch..."
      git checkout $branch

      # Rebase on latest main if not main
      if [ "$branch" != "main" ]; then
        git rebase main || {
          echo "❌ Rebase failed. Resolve conflicts manually."
          exit 1
        }
      fi

      # Push updates to both remotes
      git push origin $branch
      git push bitbucket $branch

      echo "✅ $branch synced successfully!"
      echo "----------------------------------------"
    fi
  fi
done

# Return to original branch
git checkout -
