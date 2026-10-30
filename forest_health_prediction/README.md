# HOWTO

1. Download the EDFA data

- [European Forest Disturbance Atlas (EFDA)](https://zenodo.org/records/21722408)

2. Clone `AnySat` and install locally in editable mode

- AnySat: `https://github.com/gastruc/AnySat`

```sh
uv pip install -e /path/to/your/anysat
```

or 

```sh
uv add --editable /path/to/your/anysat
```

3. Preprocess the EFDA into patches

`uv run src/preprocess.py`


