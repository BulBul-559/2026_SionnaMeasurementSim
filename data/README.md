# Data Placeholder

This directory is intentionally kept as a lightweight placeholder.

Large scene files, generated labels, floor-plan stacks, measurements, and HDF5 outputs should not be committed to this repository. Put them here locally, mount them from shared storage, or create ignored symlinks such as:

```bash
ln -s /data/sunmeiyuan/projects/sionna/scenes/bistro_0000 data/bistro_0000
ln -s /data/sunmeiyuan/projects/sionna/scenes/bistro_0001 data/bistro_0001
ln -s /data/sunmeiyuan/projects/sionna/scenes/bistro_0002 data/bistro_0002
```

Small test fixtures live under `tests/fixtures/` so the test suite does not depend on production data being present.
