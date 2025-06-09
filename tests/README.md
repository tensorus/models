# Tests

For installation instructions, see the [main README](../README.md).

To run the test suite you must have all dependencies installed, including the core
`tensorus` project that these models depend on. You can satisfy this in one of two
ways:

1. Clone the `tensorus` repository as a sibling directory next to this one so that the
   layout looks like:

   ```text
   your-workspace/
   ├─ tensorus/
   └─ models/
   ```

2. Alternatively, install the `tensorus` package beforehand (for example via
   `pip install tensorus`).

Once the dependencies are available, install the remaining requirements and run:

```bash
pip install -r ../requirements.txt
pytest
```

Some tests set the `TENSORUS_MINIMAL_IMPORT` environment variable to reduce heavy imports.
You can export it manually before running `pytest` if needed:

```bash
export TENSORUS_MINIMAL_IMPORT=1
```

The suite relies on small synthetic data via `torchvision.datasets.FakeData`, so no dataset download is required.

