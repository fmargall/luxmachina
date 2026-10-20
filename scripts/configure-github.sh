#!/usr/bin/env bash
# Configure the settings of the GitHub repository that the rulesets of
# .github/rulesets/ do not cover, then create or update those rulesets.
#
# Every change of main goes through a pull request, whose CI must pass, and is
# merged with "Squash and merge" or "Rebase and merge"; version tags can never be
# moved nor deleted. The script can be run again at any time: it sets the same
# values, and updates the rulesets that already exist instead of duplicating them.
#
# Requires the GitHub CLI (gh), authenticated as an admin of the repository.
# Usage: bash scripts/configure-github.sh [--dry-run]
#   --dry-run  only print the changes, without making any
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; fi
cd "$(dirname "$0")/.."

if ! command -v gh > /dev/null; then
    echo "error: the GitHub CLI (gh) is not installed: see https://cli.github.com/" >&2
    echo "       (on Windows: winget install --id GitHub.cli, then open a new terminal)" >&2
    echo "       On Windows, run this script with Git Bash: the bash command of PowerShell" >&2
    echo "       may be the one of WSL, which does not see the programs installed on Windows" >&2
    exit 1
fi
if ! gh auth status > /dev/null 2>&1; then
    echo "error: the GitHub CLI is not authenticated: run gh auth login first" >&2
    exit 1
fi
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
echo "Configuring $repo$($dry_run && echo ' (dry run: nothing is changed)')"

# api METHOD PATH [JSON body on stdin]: call the REST API of the repository.
# The endpoints given to gh have no leading slash: Git Bash on Windows rewrites the
# arguments starting with "/" as file paths ("/repos" -> "C:/Program Files/Git/repos")
api() {
    local method=$1 path=$2 body=""
    if [[ ! -t 0 ]]; then body=$(cat); fi
    if $dry_run; then
        echo "- would call: $method /repos/$repo$path${body:+ with $(tr -d '\n' <<< "$body" | tr -s ' ')}"
    elif [[ -n $body ]]; then
        gh api --method "$method" "repos/$repo$path" --input - <<< "$body" > /dev/null
        echo "- done: $method /repos/$repo$path"
    else
        gh api --method "$method" "repos/$repo$path" --silent < /dev/null
        echo "- done: $method /repos/$repo$path"
    fi
}

echo "Features, merge buttons and secret scanning"
# Only squash and rebase merges, as in the main ruleset. The title of the pull
# request becomes the commit message of a squash, as pull_request_template.md says
api PATCH "" <<'EOF'
{
    "has_issues": true,
    "has_wiki": false,
    "has_projects": false,
    "has_discussions": false,
    "allow_merge_commit": false,
    "allow_squash_merge": true,
    "allow_rebase_merge": true,
    "squash_merge_commit_title": "PR_TITLE",
    "squash_merge_commit_message": "PR_BODY",
    "delete_branch_on_merge": true,
    "allow_update_branch": true,
    "security_and_analysis": {
        "secret_scanning": {"status": "enabled"},
        "secret_scanning_push_protection": {"status": "enabled"}
    }
}
EOF

echo "Security"
# The private reports that SECURITY.md asks for
api PUT /private-vulnerability-reporting < /dev/null
# Dependabot alerts, and the pull requests that fix them
api PUT /vulnerability-alerts < /dev/null
api PUT /automated-security-fixes < /dev/null
# A published release, and its files signed with Sigstore, can no longer be changed
api PUT /immutable-releases < /dev/null

echo "GitHub Actions"
# Actions must be pinned to a full commit SHA, as zizmor also checks
api PUT /actions/permissions <<'EOF'
{"enabled": true, "allowed_actions": "all", "sha_pinning_required": true}
EOF
# The GITHUB_TOKEN can only read by default: each workflow and job declares the
# permissions it needs, and no workflow can create or approve pull requests
api PUT /actions/permissions/workflow <<'EOF'
{"default_workflow_permissions": "read", "can_approve_pull_request_reviews": false}
EOF

if [[ -f .github/workflows/release.yml ]]; then
    echo "Deployment environments of release.yml"
    # Only the version tags can deploy to TestPyPI and PyPI: a workflow run from
    # any branch cannot publish, even if the workflow is modified
    for environment in testpypi pypi; do
        api PUT "/environments/$environment" <<'EOF'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
EOF
        existing=$(gh api "repos/$repo/environments/$environment/deployment-branch-policies" \
            --jq '.branch_policies[] | select(.name == "v*" and .type == "tag") | .id' 2> /dev/null || true)
        if [[ -n $existing ]]; then
            echo "- the $environment environment already deploys from the v* tags only"
        else
            api POST "/environments/$environment/deployment-branch-policies" <<'EOF'
{"name": "v*", "type": "tag"}
EOF
        fi
    done
fi

echo "Rulesets of .github/rulesets"
for file in .github/rulesets/*.json; do
    name=$(sed -n 's/^  "name": "\(.*\)",$/\1/p' "$file")
    id=$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")
    if [[ -n $id ]]; then
        api PUT "/rulesets/$id" < "$file"
    else
        api POST /rulesets < "$file"
    fi
done

echo "Done. The services outside of GitHub (the trusted publishers of TestPyPI and"
echo "PyPI, Zenodo, Codecov) are configured on their own websites."
