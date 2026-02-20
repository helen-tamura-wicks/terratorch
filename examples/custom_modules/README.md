# Custom Modules Example

This example shows how to register and use your own backbone module with TerraTorch.

## Folder Layout

Place files like this:

```text
examples/custom_modules/
├── config.yaml
└── custom_modules/
    ├── __init__.py
    └── hello_geo_module.py
```

## Why `__init__.py` matters

TerraTorch imports the package from `custom_modules_path`. Registration decorators run only when the module is imported.

`custom_modules/__init__.py` must import your module class so it is registered:

```python
from custom_modules.hello_geo_module import HelloGeoModule

__all__ = ["HelloGeoModule"]
```

If this import is missing, TerraTorch cannot instantiate `backbone: HelloGeoModule`.

## Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your environment already exists, just activate it:

```bash
source .venv/bin/activate
```

## Run the Example

From this directory (`examples/custom_modules`):

```bash
terratorch fit -c ./config.yaml
```

## Key Config Values

`config.yaml` uses:

- `custom_modules_path: ./custom_modules`
- `model.init_args.model_factory: EncoderDecoderFactory`
- `model.init_args.model_args.backbone: HelloGeoModule`

This tells TerraTorch to import your package and resolve `HelloGeoModule` from the backbone registry.
