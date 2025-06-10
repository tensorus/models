# Tensorus Model Tests

This document provides details specific to running the test suite for `tensorus-models`. For general installation of the `tensorus-models` package itself, please refer to the [main README.md](../README.md#installation).

## Test Dependencies

Running the tests has a few key dependencies:

1.  **Core `tensorus` Library:**
    The models in this package rely on the core `tensorus` library (e.g., for `tensorus.tensor_storage`). You need to make this library available in your Python environment. The recommended ways are:
    *   **Install from PyPI:** `pip install tensorus`
    *   **Editable install from a local clone:** If you have a local clone of the `tensorus` repository, navigate to its root directory and run `pip install -e .`. This is useful if you're developing both libraries simultaneously.
    *(The main `README.md`'s "Running Tests" section also outlines these options).*

2.  **Python Packages for Testing:**
    Install the necessary Python packages for testing (including `pytest` and other dependencies listed in `requirements.txt` at the root of this repository):
    ```bash
    pip install -r ../requirements.txt
    ```
    *(Ensure you are in the `tests` directory or adjust the path to `requirements.txt` accordingly if running from elsewhere, though typically tests are run from the repo root).*

## Running Tests (General)

Once all dependencies are set up, you can run the entire test suite using `pytest` from the root directory of this (`tensorus-models`) repository:

```bash
pytest
```

## Running Specific Tests

While `pytest` run from the repository root will execute all tests, you can also run specific tests or groups of tests for more focused feedback during development. Here are some common ways:

*   **Run all tests in a specific file:**
    ```bash
    pytest tests/test_cnn_models.py
    ```

*   **Run all tests in a specific class within a file:**
    ```bash
    pytest tests/test_cnn_models.py::TestAlexNetModel
    ```
    *(Replace `TestAlexNetModel` with an actual class name from the test file.)*

*   **Run a specific test function within a file:**
    ```bash
    pytest tests/test_cnn_models.py::TestAlexNetModel::test_fit_predict
    ```
    *(Replace `TestAlexNetModel::test_fit_predict` with actual class and test method names.)*

*   **Run tests based on keywords in their names (across all files):**
    The `-k` option allows you to specify an expression. For example, to run all tests containing "alexnet" in their name:
    ```bash
    pytest -k "alexnet"
    ```
    Or to run tests that have "fit" but not "transformer":
    ```bash
    pytest -k "fit and not transformer"
    ```

*   **Run tests based on markers:**
    If tests are decorated with `pytest` markers (e.g., `@pytest.mark.slow`, `@pytest.mark.integration`), you can run specific groups:
    ```bash
    pytest -m slow
    ```
    *(Check the test files for any defined markers. This is an example; actual markers may vary.)*

For more advanced `pytest` usage and options, please refer to the [official pytest documentation](https://docs.pytest.org/en/stable/).

## Test Organization

Tests are located within the `tests/` directory. The general organization is as follows:

*   **By Model Category:** Most tests are grouped into files corresponding to the major model categories, such as:
    *   `test_cnn_models.py` (for Convolutional Neural Networks)
    *   `test_rl_models.py` (for Reinforcement Learning models)
    *   `test_transformer_models.py` (for Transformer-based models)
    *   `test_boosting_models.py` (for boosting algorithms)
    *   And so on for other categories like anomaly detection, time series, etc.
*   **Specific Model Tests:** Some individual complex models or detectors might have their own test files (e.g., `test_yolov5_detector.py`).
*   **General Model Tests:** `test_models.py` may contain tests applicable to the base model functionalities or a variety of models not fitting into a specific category file.

Test functions within these files typically follow standard `pytest` naming conventions (e.g., `test_*`).

## Optimizing Test Runs with `TENSORUS_MINIMAL_IMPORT`

Some parts of the `tensorus` ecosystem, including `tensorus-models`, might import large libraries (e.g., `torch`, `transformers`) at the module level. This can sometimes slow down test discovery or the execution of tests that don\'t specifically require these heavy dependencies.

To address this, some tests, and potentially the core library itself, may respect the `TENSORUS_MINIMAL_IMPORT` environment variable.

**Purpose:**
Setting `TENSORUS_MINIMAL_IMPORT=1` signals to the codebase to attempt to reduce its import footprint. This might involve:
*   Skipping the automatic import of certain large third-party libraries.
*   Enabling lazy-loading mechanisms for some components.

**When to Use:**
*   You might use this if you are running tests that you know do not depend on the full suite of heavy libraries, potentially speeding up your test cycle.
*   Some specific tests might require this variable to be set to run correctly or to test specific minimal import behaviors.

**How to Use:**
You can set this environment variable in your shell before running `pytest`:
```bash
export TENSORUS_MINIMAL_IMPORT=1
pytest
```
Or, set it for a single command:
```bash
TENSORUS_MINIMAL_IMPORT=1 pytest
```

**Note:** Be aware that running tests with `TENSORUS_MINIMAL_IMPORT=1` might cause tests that *do* require the full imports to fail if those imports are then skipped. It\'s primarily a tool for focused, faster testing of components that operate in a "minimal import" mode or for specific tests designed around this behavior. For a full, comprehensive test run ensuring all functionalities, you would typically run `pytest` without this variable set, unless specific test suites guide otherwise.

## Writing New Tests

Contributions that include new models or modify existing ones should ideally be accompanied by relevant tests. Here are some general guidelines:

1.  **Placement:**
    *   If your new model fits into an existing category file (e.g., a new CNN model), add your tests to the corresponding file (e.g., `tests/test_cnn_models.py`).
    *   If your model is distinct or complex enough, you might create a new test file (e.g., `tests/test_my_new_model.py`).
    *   Follow the existing naming convention (`test_*.py`).

2.  **Test Case Structure:**
    *   Use `pytest` conventions. Test functions should be named `test_*`.
    *   If using classes to group tests for a specific model, name the class `Test<ModelName>` (e.g., `TestMyNewModel`).

3.  **What to Test:**
    For a new model, tests should ideally cover:
    *   **Instantiation:** Can the model be initialized correctly with default and various valid parameters?
    *   **Data Handling:** Does the model correctly handle expected input data types and shapes (e.g., NumPy arrays, PyTorch tensors)?
    *   **`fit()` method:** Runs without errors on simple dummy data and shows signs of learning if applicable.
    *   **`predict()` method:** Runs without errors after `fit()`, produces correct output shapes and types.
    *   **`save()` and `load()` methods:** Model can be saved and loaded, and the loaded model produces consistent predictions.
    *   **Edge Cases (Optional):** Consider invalid inputs or parameters.

4.  **Test Data:**
    *   Use small, synthetic datasets (e.g., via `numpy.random`, `torchvision.datasets.FakeData`). Avoid external downloads in tests.

5.  **Test Independence:** Ensure tests do not depend on each other.

6.  **Fixtures:** Utilize `pytest` fixtures for repeated setup code. Check existing tests or `conftest.py` (if any) for examples.

## Test Coverage

Maintaining good test coverage is important for ensuring the quality and reliability of the codebase.

*(Note to developers: Please update this section based on the current testing practices.)*

**Current Status:**
*   (Describe if test coverage is currently being measured, e.g., "Test coverage is measured using `pytest-cov`.")
*   (If measured, state the current approximate coverage percentage if known, or where to find the latest report.)

**Running Coverage Reports:**
*   If `pytest-cov` or a similar tool is integrated, you can typically generate a coverage report by running:
    ```bash
    pytest --cov=tensorus.models --cov-report=html # Or specific path to models
    ```
    This command (if configured) would generate an HTML report in a `htmlcov/` directory.

**Goals:**
*   We aim to achieve and maintain a high level of test coverage.
*   New contributions should include tests that maintain or increase coverage.

If coverage tools are not yet set up, this is a desirable enhancement for the future.

---
*The test suite relies on small synthetic data (e.g., via `torchvision.datasets.FakeData`), so no dataset download is required for most tests.*
