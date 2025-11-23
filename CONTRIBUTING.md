# Contributing to Insulin Resistance Prediction System

Thank you for your interest in contributing! Follow these guidelines to maintain code quality and consistency.

## Branch Strategy

- **main/master:** Production-ready code only
- **develop:** Integration branch for next release
- **feature/name:** Feature development
- **bugfix/name:** Bug fixes
- **docs/name:** Documentation updates

## Development Setup

```bash
git clone <repo-url>
cd "ir prediction"
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r config/requirements.txt
```

## Before Pushing

1. **Run tests:**
   ```bash
   python scripts/run_tests.py -v
   ```

2. **Format code** (optional but encouraged):
   ```bash
   pip install black
   black src/ tests/
   ```

3. **Check for linting issues:**
   ```bash
   pip install flake8
   flake8 src/ tests/ --max-line-length=120
   ```

## Pull Request Process

1. Create feature branch from `develop`
2. Make changes, commit with clear messages
3. Push and create PR with description
4. Ensure CI passes (GitHub Actions)
5. Request review from maintainers
6. Merge to `develop` on approval

## Code Style

- **Python:** PEP 8, 120-char line length
- **Docstrings:** Google-style for all public functions
- **Type hints:** Use type annotations for function signatures

## Commit Messages

Format: `<type>: <description>`

Examples:
- `feat: add SHAP-based explainability to API`
- `fix: correct KS test drift detection threshold`
- `docs: update deployment runbook`
- `test: add robustness tests for preprocessing`

## Issues & Discussions

- Report bugs with reproduction steps in GitHub Issues
- Discuss major features in Discussions before implementing
- Reference issue numbers in PR descriptions

---

Thank you for contributing! 🙏
