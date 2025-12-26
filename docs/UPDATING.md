# How to Update Your Local Repository

When I push changes to fix bugs or add features, here's how to update your local copy:

## Quick Update (Most Common)

```bash
# 1. Navigate to your project directory
cd Fish-CNN

# 2. Pull the latest changes
git pull origin claude/fish-health-monitoring-system-MKiQ8

# 3. Update dependencies (if requirements changed)
pip install --upgrade -r requirements.txt
```

That's it! You'll now have the latest code.

## If You Have Uncommitted Changes

If you've made your own changes and want to keep them:

### Option A: Stash Your Changes (Recommended)

```bash
# 1. Save your changes temporarily
git stash

# 2. Pull latest changes
git pull origin claude/fish-health-monitoring-system-MKiQ8

# 3. Reapply your changes
git stash pop
```

### Option B: Commit Your Changes First

```bash
# 1. Commit your changes
git add .
git commit -m "My custom changes"

# 2. Pull and merge
git pull origin claude/fish-health-monitoring-system-MKiQ8

# 3. Resolve any conflicts if they arise
```

## If You Get Conflicts

If git says there are conflicts:

```bash
# 1. See which files have conflicts
git status

# 2. Open conflicted files and look for:
<<<<<<< HEAD
Your changes
=======
New changes from update
>>>>>>>

# 3. Edit to keep what you want, then:
git add <resolved-file>
git commit -m "Resolved merge conflicts"
```

## Start Fresh (Nuclear Option)

If things get messy and you want to start over:

```bash
# WARNING: This deletes all your local changes!

# 1. Navigate to parent directory
cd ..

# 2. Delete old directory
rm -rf Fish-CNN

# 3. Clone fresh copy
git clone https://github.com/yourusername/Fish-CNN.git
cd Fish-CNN

# 4. Checkout the branch
git checkout claude/fish-health-monitoring-system-MKiQ8

# 5. Reinstall
pip install -r requirements.txt
```

## Check What Changed

To see what's new in the update:

```bash
# See commit history
git log --oneline -10

# See what changed in a specific file
git diff HEAD~1 src/health_monitor.py
```

## Common Scenarios

### Scenario 1: Just Started, No Changes
```bash
git pull origin claude/fish-health-monitoring-system-MKiQ8
```

### Scenario 2: Made Config Changes Only
```bash
# Stash your config
git stash

# Pull updates
git pull origin claude/fish-health-monitoring-system-MKiQ8

# Reapply your config
git stash pop
```

### Scenario 3: Modified Code Files
```bash
# See what you changed
git status
git diff

# If you want to keep changes:
git add .
git commit -m "My modifications"
git pull origin claude/fish-health-monitoring-system-MKiQ8

# If you want to discard changes:
git reset --hard HEAD
git pull origin claude/fish-health-monitoring-system-MKiQ8
```

## After Updating

Always test that everything still works:

```bash
# 1. Test imports
python -c "from src.health_monitor import FishHealthMonitor; print('OK')"

# 2. Run a quick test
python examples/mac_camera_monitoring.py
```

## Pro Tips

1. **Before updating**, check if you have uncommitted changes:
   ```bash
   git status
   ```

2. **Create a backup** before major updates:
   ```bash
   cp -r Fish-CNN Fish-CNN-backup
   ```

3. **Update regularly** to get bug fixes and new features

4. **Read the commit messages** to see what changed:
   ```bash
   git log --oneline -5
   ```

## Get Update Notifications

To see if there are new updates available:

```bash
# Fetch latest info without downloading
git fetch origin

# See if your branch is behind
git status
```

You'll see something like:
```
Your branch is behind 'origin/claude/fish-health-monitoring-system-MKiQ8' by 2 commits
```

## Troubleshooting Updates

### Error: "Your local changes would be overwritten"
```bash
git stash
git pull origin claude/fish-health-monitoring-system-MKiQ8
git stash pop
```

### Error: "You have divergent branches"
```bash
# Option 1: Rebase (cleaner)
git pull --rebase origin claude/fish-health-monitoring-system-MKiQ8

# Option 2: Merge (safer)
git pull origin claude/fish-health-monitoring-system-MKiQ8
```

### Error: "Cannot pull with merge conflicts"
```bash
# Abort the merge
git merge --abort

# Or commit your changes first
git add .
git commit -m "My changes before update"
git pull origin claude/fish-health-monitoring-system-MKiQ8
```

## Questions?

- Check `git status` to see current state
- Use `git log` to see what's changed
- Create an issue on GitHub if stuck
