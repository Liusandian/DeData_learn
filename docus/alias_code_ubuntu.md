一、Full Git Alias Configuration (Copy to .bashrc)
# ================== Git Common Alias Configuration ==================

# Basic Operations

alias gst='git status'          # Check working directory status (most used, shortest alias)

alias gadd='git add .'          # Stage all changes in current directory (including new files)

alias gcm='git commit -m'       # Commit with message (usage: gcm "commit message")

alias gps='git push'            # Push local branch to remote repository

alias gpl='git pull'            # Pull latest changes from remote repository

alias gbr='git branch -a'       # List all branches (local + remote; remote starts with "remotes/")

alias gch='git checkout'        # Switch to specified branch (usage: gch branch-name)

alias gchb='git checkout -b'    # Create and switch to new branch (usage: gchb new-branch-name)

# Stash Series

alias gsts='git stash'          # Stash working directory changes (temporary save without commit)

alias gstsl='git stash list'    # List all stash records (with index + description)

alias gstsap='git stash apply'  # Apply latest stash (keep stash record)

alias gstsapn='git stash apply stash@{n}' # Apply specified stash (n = index; usage: gstsapn 0)

alias gstsp='git stash pop'     # Apply latest stash and delete the record

alias gstsd='git stash drop'    # Delete latest stash (usage: gstsd or gstsd stash@{n})

alias gstscl='git stash clear'  # Clear all stash records

# Rebase Series (Newly Added)

alias grebase='git rebase'      # Basic rebase (usage: grebase target-branch)

alias grebasei='git rebase -i'  # Interactive rebase (for squashing/editing commits; usage: grebasei HEAD\~3)

alias grebasec='git rebase --continue' # Continue rebase after resolving conflicts

alias grebasea='git rebase --abort'   # Abort rebase (discard changes)

alias grebases='git rebase --skip'    # Skip current commit during rebase

# Combined Commands (Newly Added: Merge Multiple Operations)

alias gac='git add . && git commit -m' # Stage all + commit (usage: gac "commit message")

alias gacp='git add . && git commit -m "\$1" && git push' # Stage + commit + push (usage: gacp "commit message")

alias gplr='git pull --rebase'  # Pull with rebase (avoid merge commits; usage: gplr origin main)

alias gbrd='git branch -d'      # Delete local branch (usage: gbrd branch-name)

alias gbrdd='git branch -D'     # Force delete local branch (usage: gbrdd branch-name)

alias gpushnew='git push -u origin' # Push new branch to remote (usage: gpushnew new-branch-name)

# Auxiliary Operations

alias glog='git log --oneline --graph' # Concise commit history (one line per commit + branch graph)

alias gdiff='git diff'          # View file modification differences

alias gmerge='git merge'        # Merge target branch (usage: gmerge target-branch)

alias gremote='git remote -v'   # View remote repository addresses

alias gclean='git clean -fd'    # Delete untracked files/directories (use with caution)

alias greset='git reset --hard' # Force reset to latest commit (use with extreme caution)

# ========================================================
二、Newly Added Alias Reference Table
Short Alias	Corresponding Original Git Command	Core Function Description
grebase	git rebase	Rebase current branch onto target branch (e.g., grebase main)
grebasei	git rebase -i	Interactive rebase (edit/squash/reorder commits; e.g., grebasei HEAD~3 for last 3 commits)
grebasec	git rebase --continue	Resume rebase after resolving merge conflicts
grebasea	git rebase --abort	Cancel rebase and restore branch to original state
grebases	git rebase --skip	Skip current problematic commit during rebase
gac	git add . && git commit -m	One-step: Stage all changes + commit (e.g., gac "fix: optimize login logic")
gacp	git add . && git commit -m "$1" && git push	One-step: Stage + commit + push (e.g., gacp "feat: add user profile page")
gplr	git pull --rebase	Pull remote changes and rebase (avoids extra merge commits)
gbrd	git branch -d	Delete local branch (only if merged to main)
gbrdd	git branch -D	Force delete local branch (even if unmerged)
gpushnew	git push -u origin	Push new local branch to remote and set upstream (e.g., gpushnew feature/payment)
三、Usage Notes for New Aliases
Rebase Series:
Use grebasei HEAD~n to modify the last n commits (e.g., grebasei HEAD~5 for 5 commits) – ideal for squashing multiple small commits into one before pushing.

If conflicts occur during rebase: resolve conflicts → gadd . → grebasec to continue, or grebasea to abort.

Combined Commands:
gacp requires a commit message (enclosed in quotes) – it’s the fastest way to push small changes.

gplr is recommended over gpl for feature branches (keeps commit history linear).

gclean deletes untracked files (e.g., build folders, logs) – run gst first to confirm no important untracked files.

Permanent Activation:

After pasting the full configuration into ~/.bashrc, run:

source \~/.bashrc
All new aliases will take effect immediately.

（注：文档部分内容可能由 AI 生成）