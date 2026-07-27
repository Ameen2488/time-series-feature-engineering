# Contributing

Thanks for your interest in improving this repository.

## Reporting Issues

If you've found a bug in the code, a factual error in the articles, or something
that doesn't work as documented, please open a GitHub issue with:

- What you expected to happen
- What actually happened
- Steps to reproduce (minimal example)
- Python version and OS

## Suggesting Content

If there's a time series feature engineering topic you'd like to see covered
in a future article, open a Discussion in the "Ideas" category.

## Pull Requests

Small fixes (typos, doc improvements, additional tests) are welcome without prior
discussion. For larger changes, please open an issue first to align on scope.

### Development setup

```bash
git clone https://github.com/Ameen2488/time-series-feature-engineering.git
cd time-series-feature-engineering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Before submitting

1. Ensure tests pass: `pytest tests/ -v`
2. Ensure the notebook you touched still executes: `jupyter nbconvert --to notebook --execute notebooks/XX_yourfile.ipynb --output _test.ipynb`
3. If you added new utility code in `src/`, add tests

## Code Style

- Follow PEP 8 (enforced loosely via `ruff`)
- Type hints on function signatures where it aids readability
- Docstrings in NumPy style for public functions

## Discussion & Questions

For questions about the concepts themselves (rather than the code), the best
places are:

- Comments on the [Medium articles](https://medium.com/@asidd24)
- Replies on the LinkedIn Newsletter editions
- GitHub Discussions
