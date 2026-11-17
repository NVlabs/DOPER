.PHONY: format check test coverage

# Public source paths that `format` and `check` cover.
LINT_PATHS := doper rendering tests
TYPE_PATHS := doper rendering
# Extra paths appended by an invocation, for example `make check EXTRA_PATHS=other_dir`.
EXTRA_PATHS :=
# Pytest marker expression for `test` and `coverage`.
TEST_MARKERS := not gpu and not isaac

format:
	ruff format $(LINT_PATHS) $(EXTRA_PATHS)
	ruff check --fix $(LINT_PATHS) $(EXTRA_PATHS)

check:
	ruff check $(LINT_PATHS) $(EXTRA_PATHS)
	ruff format --check $(LINT_PATHS) $(EXTRA_PATHS)
	pyright $(TYPE_PATHS) $(EXTRA_PATHS)

test:
	pytest -m "$(TEST_MARKERS)"

coverage:
	pytest -m "$(TEST_MARKERS)" --cov=doper
