# Pushing This Repository to GitHub

Once-off setup instructions for taking this local repository and publishing it to GitHub as `time-series-feature-engineering`.

**Do NOT commit this file** — delete it after the repo is live.

---

## Prerequisites

- A GitHub account
- Git installed locally
- (Optional but recommended) GitHub CLI: [`gh`](https://cli.github.com/)

---

## Option A — With GitHub CLI (fastest)

```bash
cd /path/to/ts-feature-engineering

# Initialize local git repo
git init
git add .
git commit -m "Initial commit — repo scaffolding, Article 1 notebook & utilities"

# Create GitHub repo AND push in one command
gh repo create time-series-feature-engineering \
    --public \
    --description "Production-grade feature engineering for time series forecasting and anomaly detection — companion code for a 10-part article series." \
    --source=. \
    --remote=origin \
    --push
```

## Option B — Manual (web + git)

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `time-series-feature-engineering`
3. Description: *Production-grade feature engineering for time series forecasting and anomaly detection — companion code for a 10-part article series.*
4. Public
5. **Do NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

Then locally:

```bash
cd /path/to/ts-feature-engineering
git init
git add .
git commit -m "Initial commit — repo scaffolding, Article 1 notebook & utilities"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/time-series-feature-engineering.git
git push -u origin main
```

---

## After Push — Configure Repository

### 1. Add topics (for discoverability)

Go to the repo → gear icon next to "About" → add topics:

```
time-series  forecasting  feature-engineering  machine-learning
demand-forecasting  anomaly-detection  python  scikit-learn
lightgbm  data-science  supply-chain  mlops
```

### 2. Add a repo description and website link

- Description: *Production-grade feature engineering for time series forecasting and anomaly detection*
- Website: your Medium profile or LinkedIn Newsletter URL

### 3. Enable Discussions

Settings → General → Features → check "Discussions"

Then create these categories:
- Announcements
- Ideas (for topic suggestions)
- Q&A
- Show and tell

### 4. Configure GitHub Pages (optional)

If you want the docs directory rendered as a website:
Settings → Pages → Source: main branch, /docs folder

### 5. Pin the repository on your profile

- Go to your GitHub profile
- Click "Customize your pins"
- Add `time-series-feature-engineering`

### 6. Add repo card image (recommended)

Settings → General → scroll to "Social preview" → upload a 1280x640 image.
This is what appears when the repo is shared on LinkedIn, Twitter, Medium.

---

## Update Placeholder Links

Search and replace throughout the repo before pushing:

| Placeholder | Replace with |
|---|---|
| `your-username` | Your Medium username |
| `YOUR_USERNAME` | Your GitHub username |
| `medium.com/@your-username` | Actual Medium profile URL |
| `linkedin.com/newsletters/` | Actual Newsletter URL (once created) |

Recommended one-liner (macOS/Linux):

```bash
# From the repo root, replace GitHub username
grep -rl 'your-username' . --exclude-dir=.git --exclude-dir=.venv | \
  xargs sed -i '' 's/your-username/ACTUAL_USERNAME/g'
```

---

## Verify the CI Runs

After the first push:

1. Go to the "Actions" tab on GitHub
2. You should see the CI workflow running
3. It runs tests on Python 3.10, 3.11, and 3.12
4. Add the badge to the README once it's green:
   ```markdown
   ![CI](https://github.com/YOUR_USERNAME/time-series-feature-engineering/workflows/CI/badge.svg)
   ```

---

## Link Back from Your Content

Once live, add the repo URL to:

- Your LinkedIn "Featured" section
- Your LinkedIn About section
- Your Medium profile bio
- The footer of every article: *"Full code: github.com/YOUR_USERNAME/time-series-feature-engineering"*
- Your resume (Publications or Projects section)

---

## Delete This File

```bash
rm GITHUB_SETUP.md
git add -A && git commit -m "Remove setup instructions" && git push
```
