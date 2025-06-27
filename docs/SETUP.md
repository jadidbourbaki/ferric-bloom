# GitHub Pages Setup Instructions

## Automatic Setup (Recommended)

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Scroll to "Pages" section
   - Set source to "GitHub Actions"

2. **Push the docs directory**:
   - The GitHub Actions workflow will automatically build and deploy the site
   - Site will be available at `https://yourusername.github.io/ferric-bloom/`

## Manual Setup (Alternative)

If you prefer to set up GitHub Pages manually:

1. **Enable GitHub Pages**:
   - Go to repository settings → Pages
   - Set source to "Deploy from a branch"
   - Select `main` branch and `/docs` folder

2. **Configure Jekyll**:
   - GitHub Pages will automatically detect Jekyll configuration
   - The site will build using the `_config.yml` settings

## Repository Settings Required

Make sure your repository has:
- Public visibility (or GitHub Pro for private repos)
- Pages enabled in repository settings
- Actions enabled if using automatic deployment

## Troubleshooting

- If builds fail, check the Actions tab for error logs
- Ensure all markdown files exist and are properly formatted
- Check that relative paths in `{% include_relative %}` are correct 