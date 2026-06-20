# trajlens

The quality and synthesis layer for the open robot-learning data ecosystem.

ruff for robot data — lint, fix, and generate clean LeRobotDataset datasets.

## Status

Pre-v0.1, under active development. Not yet on PyPI.

## Install (dev)

```bash
git clone https://github.com/<your-username>/trajlens
cd trajlens
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,hub]"
```

## Performance Note: Hub vs. Local

Linting a 100-episode dataset locally takes under 30 seconds.

Linting a Hub dataset directly (`trajlens lint org/dataset`) streams data over HTTP. Because it must execute isolated network requests for data shards, it will inherently be slower than a local copy (typically ~1 to 3 minutes depending on shard size and network speed). For repeated linting, we strongly recommend downloading the dataset locally first.

## License

Apache-2.0
