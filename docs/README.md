# GitHub Pages Documentation

This directory contains the Jekyll configuration for the Ferric Bloom documentation site hosted on GitHub Pages.

## Structure

- `_config.yml` - Jekyll configuration
- `Gemfile` - Ruby dependencies
- `index.md` - Home page (includes main README.md)
- `python-bindings.md` - Python bindings documentation
- `comparison.md` - Performance comparison with Blowchoc
- `testing.md` - Testing environment setup

## Local Development

To run the site locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000/ferric-bloom/`

## Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch.

## Features

- **Direct inclusion**: Markdown files are included directly without duplication
- **Automatic navigation**: Header navigation is configured in `_config.yml`
- **Responsive design**: Uses the Minima theme for clean, responsive layout
- **SEO optimization**: Includes Jekyll SEO tag plugin 