# GitHub Pages Documentation

This directory contains a simple Jekyll site that displays the Ferric Bloom README on GitHub Pages.

## Setup

1. Go to your repository settings
2. Scroll to "Pages" section
3. Set source to "Deploy from a branch"
4. Select `main` branch and `/docs` folder

The site will be available at: `https://yourusername.github.io/ferric-bloom/`

## Files

- `_config.yml` - Jekyll configuration
- `Gemfile` - Ruby dependencies
- `index.md` - Main page with README content
- `README.md` - This file

## Local Development

To run locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Site will be available at `http://localhost:4000/ferric-bloom/` 