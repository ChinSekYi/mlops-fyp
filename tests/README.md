
# Testing

This folder contains all test suites for the project, organized by type:

- `unit/` — Unit tests for individual functions and modules
- `integration/` — Integration tests for components working together
- `e2e/` — End-to-end tests for the full pipeline and API
- `smoke/` — Smoke tests for production checks

## Usage

Before running tests, ensure the required containers (MLflow, API/backend) are running for your target environment.

### Common Makefile Commands

- **All tests (with warnings ignored):**
    ```sh
    make test
    ```
- **Unit tests (local):**
    ```sh
    make test-dev
    ```
- **Integration tests (CI):**
    ```sh
    make test-pipeline-ci
    ```
- **Staging E2E tests:**
    ```sh
    make test-staging
    ```
- **Production smoke tests:**
    ```sh
    make test-prod
    ```

Or use `pytest` directly:

- Run all tests:
    ```sh
    pytest tests/
    ```
- Run specific suite:
    ```sh
    pytest -v tests/unit/
    pytest -v tests/integration/
    pytest -v tests/e2e/
    pytest -v tests/smoke/
    ```
- Run a specific test:
    ```sh
    pytest tests/unit/test_model.py::test_model_prediction_shape -v
    ```

## Pytest Notes

- **Fixtures:** Functions that set up data or objects for multiple tests, avoiding repeated setup code. Pytest injects fixtures automatically when you add them as parameters to your test functions.
