# Tests

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

Once the dependencies are available, install the remaining requirements and run
`pytest`:

```bash
pip install -r ../requirements.txt
pytest
```
