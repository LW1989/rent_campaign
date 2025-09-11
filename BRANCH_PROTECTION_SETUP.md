# GitHub Branch Protection Setup Guide

This guide will help you configure branch protection rules for the rent_campaign repository to ensure code quality and prevent broken code from reaching the main branch.

## Prerequisites

- You must have admin access to the GitHub repository
- The CI/CD workflows must be set up (already done ✅)
- At least one successful workflow run should have completed

## Step-by-Step Setup

### 1. Navigate to Repository Settings

1. Go to your repository: https://github.com/LW1989/rent_campaign
2. Click on **Settings** tab (requires admin access)
3. In the left sidebar, click **Branches**

### 2. Add Branch Protection Rule

1. Click **Add rule** button
2. In the **Branch name pattern** field, enter: `main`

### 3. Configure Protection Settings

#### Required Status Checks ✅
- [x] **Require status checks to pass before merging**
- [x] **Require branches to be up to date before merging**

**Select these status checks:**
- `Fast Tests (Python 3.11)` - from CI workflow
- `Integration Tests (Python 3.11)` - from CI workflow  
- `Code Quality` - from CI workflow
- `PR Status Check` - from PR validation workflow

#### Pull Request Requirements ✅
- [x] **Require a pull request before merging**
- [x] **Require approvals: 1** (adjust based on team size)
- [x] **Dismiss stale pull request approvals when new commits are pushed**
- [x] **Require review from code owners** (optional, if you have CODEOWNERS file)

#### Additional Restrictions ✅
- [x] **Restrict pushes that create files that are larger than 100 MB**
- [x] **Include administrators** (applies rules to admin users too)
- [x] **Allow force pushes: No one** (prevents force pushes)
- [x] **Allow deletions: No one** (prevents branch deletion)

### 4. Save Protection Rule

Click **Create** to save the branch protection rule.

## What This Achieves

### 🛡️ Quality Gates
- **No direct pushes to main**: All changes must go through pull requests
- **Automated testing**: All tests must pass before merging
- **Code review**: At least one approval required
- **Up-to-date branches**: PRs must be current with main branch

### 🔄 Development Workflow
```
Developer workflow:
1. Create feature branch: git checkout -b feature/new-feature
2. Make changes and commit
3. Push branch: git push origin feature/new-feature
4. Create Pull Request on GitHub
5. CI/CD runs automatically (fast tests, integration tests, code quality)
6. Request review from team member
7. Address any feedback or test failures
8. Once approved and tests pass → Merge allowed
```

### 📊 Status Checks Required

The following checks must pass before merging:

| Check | Description | Timeout |
|-------|-------------|---------|
| Fast Tests | Unit tests (<30s) | 5 min |
| Integration Tests | Workflow tests (<5min) | 10 min |
| Code Quality | Linting, formatting | 5 min |
| PR Status Check | Comprehensive PR validation | 15 min |

## Troubleshooting

### Issue: Status checks not appearing
**Solution:** 
1. Ensure CI workflows have run at least once successfully
2. The status check names must match exactly with workflow job names
3. Wait a few minutes after workflow completion for GitHub to recognize them

### Issue: "Include administrators" is too restrictive
**Solution:**
- You can temporarily disable this during initial setup
- Re-enable once the team is comfortable with the workflow

### Issue: Too many required approvals
**Solution:**
- Start with 1 approval for small teams
- Increase to 2+ for larger teams or critical repositories

## Testing the Setup

### 1. Create a Test PR
```bash
# Create test branch
git checkout -b test-branch-protection

# Make a small change
echo "# Test change" >> test_file.md
git add test_file.md
git commit -m "Test branch protection"

# Push and create PR
git push origin test-branch-protection
```

### 2. Verify Protection Works
1. Try to push directly to main (should be blocked)
2. Create PR from test branch
3. Verify CI checks run automatically
4. Verify merge is blocked until checks pass
5. Verify approval is required

### 3. Clean Up
Delete the test branch and PR after verification.

## Advanced Configuration (Optional)

### Code Owners File
Create `.github/CODEOWNERS` to automatically request reviews:

```
# Global owners
* @LW1989

# Specific paths
src/ @LW1989
tests/ @LW1989  
.github/ @LW1989
```

### Required Checks by File Type
Add path-based protection for sensitive files:

```
# In branch protection settings, you can add path-based restrictions
# Example: require additional reviews for workflow changes
.github/workflows/ - require 2 approvals
src/functions.py - require code owner review
```

## Maintenance

### Monthly Review
- Check if new workflow jobs should be added to required status checks
- Review approval requirements based on team growth
- Update timeout settings if needed

### When Adding New Workflows
1. Add new status check names to branch protection rules
2. Test with a draft PR first
3. Update this documentation

---

## Quick Reference

**Minimum Recommended Settings:**
- ✅ Require PR before merging  
- ✅ Require 1 approval
- ✅ Require status checks: Fast Tests, Integration Tests, Code Quality
- ✅ Include administrators
- ✅ No force pushes

**Repository URL:** https://github.com/LW1989/rent_campaign/settings/branches

**Status:** Ready to configure ✅
