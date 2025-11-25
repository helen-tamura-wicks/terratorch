import gc
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
import torch.nn as nn

from terratorch.tasks.embedding_generation import EmbeddingGenerationTask


# ---- Fixtures ----

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_model():
    """Create a simple mock model for testing."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    # Return a simple tensor as output
    model.return_value = torch.randn(2, 64, 7, 7)
    return model


@pytest.fixture
def vit_model():
    """Create a mock ViT model that returns (B, N, D) embeddings."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    # Return token embeddings: batch=2, tokens=197 (196+1 CLS), dim=768
    model.return_value = torch.randn(2, 197, 768)
    return model


@pytest.fixture
def multimodal_model():
    """Create a mock model that returns dict of modalities."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    # Return dict with different modalities
    model.return_value = {
        'optical': torch.randn(2, 64, 7, 7),
        'radar': torch.randn(2, 32, 7, 7)
    }
    return model


@pytest.fixture
def multilayer_model():
    """Create a mock model that returns multiple layer outputs."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    # Return list of layer outputs
    model.return_value = [
        torch.randn(2, 32, 14, 14),
        torch.randn(2, 64, 7, 7),
        torch.randn(2, 128, 7, 7)
    ]
    return model


# ---- Initialization Tests ----

def test_task_initialization_defaults(temp_dir):
    """Test task initialization with default parameters."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(
            model="test_model",
            output_dir=temp_dir
        )
        # model string is replaced by actual model during configure_models
        assert task.output_path == Path(temp_dir)
        assert task.layers == [-1]
        assert task.output_format == "tiff"
        assert task.embed_file_key == "filename"
    gc.collect()


def test_task_initialization_custom_params(temp_dir):
    """Test task initialization with custom parameters."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(
            model="test_model",
            output_dir=temp_dir,
            layers=[0, -1, -2],
            output_format="parquet",
            embed_file_key="file_id",
            has_cls=True,
            embedding_pooling="vit_mean"
        )
        assert task.layers == [0, -1, -2]
        assert task.output_format == "parquet"
        assert task.embed_file_key == "file_id"
        assert task.has_cls is True
        assert task.embedding_pooling == "vit_mean"
    gc.collect()


def test_task_initialization_invalid_format(temp_dir):
    """Test that invalid output format raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        with pytest.raises(ValueError, match="Unsupported output format"):
            EmbeddingGenerationTask(
                model="test_model",
                output_dir=temp_dir,
                output_format="invalid"
            )
    gc.collect()


def test_task_initialization_pooling_warnings(temp_dir):
    """Test that appropriate warnings are raised for pooling configurations."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        
        # Test warning when has_cls not provided with vit pooling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task = EmbeddingGenerationTask(
                model="test_model",
                output_dir=temp_dir,
                embedding_pooling="vit_mean"
            )
            assert any("No 'has_cls' provided" in str(warning.message) for warning in w)
        
        # Test warning for GeoTIFF with pooling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task = EmbeddingGenerationTask(
                model="test_model",
                output_dir=temp_dir,
                output_format="tiff",
                embedding_pooling="vit_mean",
                has_cls=True
            )
            assert any("GeoTIFF output not recommended" in str(warning.message) for warning in w)
    gc.collect()


def test_task_with_temporal_wrapper(temp_dir):
    """Test task initialization with temporal wrapper configuration."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(
            model="test_model",
            output_dir=temp_dir,
            temporal_cfg={"temporal_wrapper": True, "temporal_pooling": "mean"}
        )
        assert task.temporal_cfg["temporal_wrapper"] is True
        assert task.temporal_cfg["temporal_pooling"] == "mean"
    gc.collect()


# ---- Shape Inference Tests ----

def test_infer_bt_4d_input():
    """Test BT inference from 4D input [B, C, H, W]."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 224, 224)
        B, T = task.infer_BT(x)
        assert B == 2
        assert T == 1
    gc.collect()


def test_infer_bt_5d_input():
    """Test BT inference from 5D input [B, C, T, H, W]."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 4, 224, 224)
        B, T = task.infer_BT(x)
        assert B == 2
        assert T == 4
    gc.collect()


def test_infer_bt_dict_input():
    """Test BT inference from dict input."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = {'optical': torch.randn(2, 3, 4, 224, 224), 'radar': torch.randn(2, 2, 4, 224, 224)}
        B, T = task.infer_BT(x)
        assert B == 2
        assert T == 4
    gc.collect()


# ---- File ID Validation Tests ----

def test_check_file_ids_valid_tensor_4d():
    """Test file_ids validation with 4D input and (B,) file_ids."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 224, 224)
        file_ids = torch.tensor([0, 1])
        task.check_file_ids(file_ids, x)  # Should not raise
    gc.collect()


def test_check_file_ids_valid_tensor_5d():
    """Test file_ids validation with 5D input and (B, T) file_ids."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 4, 224, 224)
        file_ids = torch.zeros((2, 4))
        task.check_file_ids(file_ids, x)  # Should not raise
    gc.collect()


def test_check_file_ids_valid_list():
    """Test file_ids validation with list input."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 224, 224)
        file_ids = ["file1.tif", "file2.tif"]
        task.check_file_ids(file_ids, x)  # Should not raise
    gc.collect()


def test_check_file_ids_valid_nested_list():
    """Test file_ids validation with nested list for temporal data."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 3, 224, 224)
        file_ids = [["t1.tif", "t2.tif", "t3.tif"], ["t4.tif", "t5.tif", "t6.tif"]]
        task.check_file_ids(file_ids, x)  # Should not raise
    gc.collect()


def test_check_file_ids_invalid_shape():
    """Test that invalid file_ids shape raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 224, 224)
        file_ids = torch.zeros((3,))  # Wrong batch size
        with pytest.raises(ValueError, match="file_ids.*shape mismatch"):
            task.check_file_ids(file_ids, x)
    gc.collect()


def test_check_file_ids_invalid_temporal_length():
    """Test that invalid temporal length in file_ids raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 4, 224, 224)
        file_ids = [["t1.tif", "t2.tif"], ["t3.tif", "t4.tif"]]  # T=2 but should be T=4
        with pytest.raises(ValueError, match="inner length"):
            task.check_file_ids(file_ids, x)
    gc.collect()


def test_check_file_ids_invalid_type():
    """Test that invalid file_ids type raises TypeError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(TypeError, match="must be a tensor"):
            task.check_file_ids("invalid", x)
    gc.collect()


# ---- Embedding Extraction Tests ----

def test_get_embeddings_single_layer(simple_model):
    """Test embedding extraction for single layer."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = simple_model
        task = EmbeddingGenerationTask(model="test_model", layers=[-1])
        task.configure_models()
        
        x = torch.randn(2, 3, 224, 224)
        embeddings, layers = task.get_embeddings(x, [-1])
        
        assert len(embeddings) == 1
        assert len(layers) == 1
        assert layers[0] == 0
    gc.collect()


def test_get_embeddings_multiple_layers(multilayer_model):
    """Test embedding extraction for multiple layers."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = multilayer_model
        task = EmbeddingGenerationTask(model="test_model", layers=[0, -1, -2])
        task.configure_models()
        
        x = torch.randn(2, 3, 224, 224)
        embeddings, layers = task.get_embeddings(x, [0, -1, -2])
        
        assert len(embeddings) == 3
        assert len(layers) == 3
        assert layers == [0, 2, 1]  # After negative index resolution
    gc.collect()


def test_get_embeddings_out_of_bounds():
    """Test that out of bounds layer index raises IndexError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = [torch.randn(2, 64, 7, 7)]
        mock_registry.build.return_value = model
        task = EmbeddingGenerationTask(model="test_model", layers=[5])
        task.configure_models()
        
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(IndexError, match="Layer index.*out of bounds"):
            task.get_embeddings(x, [5])
    gc.collect()


def test_get_embeddings_model_failure():
    """Test that model inference failure raises RuntimeError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.side_effect = Exception("Model forward failed")
        mock_registry.build.return_value = model
        task = EmbeddingGenerationTask(model="test_model")
        task.configure_models()
        
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(RuntimeError, match="Model inference failed"):
            task.get_embeddings(x, [-1])
    gc.collect()


# ---- Metadata Extraction Tests ----

def test_pull_metadata_all_fields():
    """Test metadata extraction with all supported fields."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        data = {
            "file_id": "id123",
            "product_id": "prod456",
            "time": "2023-01-01",
            "grid_cell": "cell789",
            "grid_row_u": 10,
            "grid_col_r": 20,
            "geometry": "POINT(0 0)",
            "utm_footprint": "footprint",
            "crs": "EPSG:4326",
            "pixel_bbox": [0, 0, 100, 100],
            "bounds": [0, 0, 1, 1],
            "center_lat": 45.0,
            "center_lon": -73.0,
            "extra_field": "should_remain"
        }
        
        metadata = task.pull_metadata(data)
        
        assert metadata["file_id"] == "id123"
        assert metadata["product_id"] == "prod456"
        assert metadata["time"] == "2023-01-01"
        assert "extra_field" not in metadata
        assert "extra_field" in data  # Original data unchanged
    gc.collect()


def test_pull_metadata_with_aliases():
    """Test metadata extraction with field aliases."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        data = {
            "time_": "2023-01-01",  # Alias for 'time'
            "centre_lat": 45.0,     # Alias for 'center_lat'
            "centre_lon": -73.0     # Alias for 'center_lon'
        }
        
        metadata = task.pull_metadata(data)
        
        assert metadata["time"] == "2023-01-01"
        assert metadata["center_lat"] == 45.0
        assert metadata["center_lon"] == -73.0
    gc.collect()


def test_pull_metadata_empty():
    """Test metadata extraction with empty input."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        data = {"some_field": "value"}
        metadata = task.pull_metadata(data)
        
        assert len(metadata) == 0
    gc.collect()


# ---- Pooling Tests ----

def test_pool_embedding_none():
    """Test that None pooling returns unchanged embedding."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)
        result = task.pool_embedding(embedding, None, None)
        assert torch.equal(result, embedding)
    gc.collect()


def test_pool_embedding_vit_mean():
    """Test ViT mean pooling."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)
        result = task.pool_embedding(embedding, "vit_mean", has_cls=True)
        assert result.shape == (768,)
        # Verify CLS was dropped
        assert torch.allclose(result, embedding[1:, :].mean(dim=0))
    gc.collect()


def test_pool_embedding_vit_max():
    """Test ViT max pooling."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)
        result = task.pool_embedding(embedding, "vit_max", has_cls=True)
        assert result.shape == (768,)
    gc.collect()


def test_pool_embedding_vit_cls():
    """Test ViT CLS pooling."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)
        result = task.pool_embedding(embedding, "vit_cls", has_cls=True)
        assert result.shape == (768,)
        assert torch.equal(result, embedding[0, :])
    gc.collect()


def test_pool_embedding_vit_cls_error_no_cls():
    """Test that vit_cls pooling without CLS raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(196, 768)
        with pytest.raises(ValueError, match="Cannot use 'vit_cls' pooling without a CLS token"):
            task.pool_embedding(embedding, "vit_cls", has_cls=False)
    gc.collect()


def test_pool_embedding_cnn_mean():
    """Test CNN mean pooling."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(64, 7, 7)
        result = task.pool_embedding(embedding, "cnn_mean", has_cls=None)
        assert result.shape == (64,)
    gc.collect()


@pytest.mark.xfail(reason="Bug in source: PyTorch max() doesn't accept tuple for dim parameter")
def test_pool_embedding_cnn_max():
    """Test CNN max pooling."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(64, 7, 7)
        result = task.pool_embedding(embedding, "cnn_max", has_cls=None)
        assert result.shape == (64,)
        # Verify it's actually the max values
        expected = embedding.max(dim=1).values.max(dim=1).values
        assert torch.allclose(result, expected)
    gc.collect()


def test_pool_embedding_invalid_vit_dims():
    """Test that ViT pooling on wrong dimensions raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(64, 7, 7)  # 3D instead of 2D
        with pytest.raises(ValueError, match="Expected 2D embedding for ViT pooling"):
            task.pool_embedding(embedding, "vit_mean", has_cls=True)
    gc.collect()


def test_pool_embedding_invalid_cnn_dims():
    """Test that CNN pooling on wrong dimensions raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)  # 2D instead of 3D
        with pytest.raises(ValueError, match="Expected 3D embedding for CNN pooling"):
            task.pool_embedding(embedding, "cnn_mean", has_cls=None)
    gc.collect()


def test_pool_embedding_unsupported_method():
    """Test that unsupported pooling method raises ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        
        embedding = torch.randn(197, 768)
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            task.pool_embedding(embedding, "invalid_pooling", has_cls=True)
    gc.collect()


# ---- GeoTIFF Writing Tests ----

def test_write_tiff_1d_embedding(temp_dir):
    """Test writing 1D embedding to GeoTIFF."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(768)
        metadata = {"time": "2023-01-01", "crs": "EPSG:4326"}
        
        task.write_tiff(embedding, "test_file", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_file_embedding.tif"
        assert output_file.exists()
        
        with rasterio.open(output_file) as src:
            assert src.count == 768
            assert src.height == 1
            assert src.width == 1
    gc.collect()


def test_write_tiff_2d_embedding(temp_dir):
    """Test writing 2D (ViT) embedding to GeoTIFF."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, has_cls=True)
        
        # 197 tokens (196 patches + 1 CLS) with 768 dims
        embedding = torch.randn(197, 768)
        metadata = {}
        
        task.write_tiff(embedding, "test_vit", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_vit_embedding.tif"
        assert output_file.exists()
        
        with rasterio.open(output_file) as src:
            # After removing CLS, 196 = 14*14
            assert src.count == 768
            assert src.height == 14
            assert src.width == 14
    gc.collect()


def test_write_tiff_3d_embedding(temp_dir):
    """Test writing 3D (CNN) embedding to GeoTIFF."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(64, 7, 7)
        metadata = {}
        
        task.write_tiff(embedding, "test_cnn", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_cnn_embedding.tif"
        assert output_file.exists()
        
        with rasterio.open(output_file) as src:
            assert src.count == 64
            assert src.height == 7
            assert src.width == 7
    gc.collect()


def test_write_tiff_with_metadata_tags(temp_dir):
    """Test that metadata is written as GeoTIFF tags."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(64)
        metadata = {"time": "2023-01-01", "product_id": "test123"}
        
        task.write_tiff(embedding, "test_meta", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_meta_embedding.tif"
        with rasterio.open(output_file) as src:
            tags = src.tags()
            assert tags["time"] == "2023-01-01"
            assert tags["product_id"] == "test123"
    gc.collect()


def test_write_tiff_vit_non_square_error(temp_dir):
    """Test that non-square ViT embeddings raise ValueError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, has_cls=True)
        
        # 200 tokens (199 patches + 1 CLS) - not a perfect square after removing CLS
        embedding = torch.randn(200, 768)
        metadata = {}
        
        with pytest.raises(ValueError, match="Cannot reshape.*tokens into.*grid"):
            task.write_tiff(embedding, "test_nonsquare", metadata, Path(temp_dir))
    gc.collect()


# ---- GeoParquet Writing Tests ----

def test_write_parquet_1d_embedding(temp_dir):
    """Test writing 1D embedding to GeoParquet."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, output_format="parquet")
        
        embedding = torch.randn(768)
        metadata = {"time": np.array("2023-01-01"), "center_lat": np.array(45.0)}
        
        task.write_parquet(embedding, "test_parquet", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_parquet_embedding.parquet"
        assert output_file.exists()
        
        import pandas as pd
        df = pd.read_parquet(output_file)
        assert len(df) == 1
        assert "embedding" in df.columns
        assert df["embedding"][0] is not None
        assert df["time"][0] == "2023-01-01"
    gc.collect()


def test_write_parquet_2d_embedding(temp_dir):
    """Test writing 2D embedding to GeoParquet."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, output_format="parquet")
        
        embedding = torch.randn(197, 768)
        metadata = {}
        
        task.write_parquet(embedding, "test_2d_parquet", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_2d_parquet_embedding.parquet"
        assert output_file.exists()
    gc.collect()


def test_write_parquet_3d_embedding(temp_dir):
    """Test writing 3D embedding to GeoParquet."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, output_format="parquet")
        
        embedding = torch.randn(64, 7, 7)
        metadata = {}
        
        task.write_parquet(embedding, "test_3d_parquet", metadata, Path(temp_dir))
        
        output_file = Path(temp_dir) / "test_3d_parquet_embedding.parquet"
        assert output_file.exists()
    gc.collect()


# ---- Batch Writing Tests ----

def test_write_batch_single_samples(temp_dir):
    """Test writing batch without temporal dimension."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(2, 64, 7, 7)
        file_ids = ["file1.tif", "file2.tif"]
        metadata = {"time": ["2023-01-01", "2023-01-02"]}
        
        task.write_batch(embedding, file_ids, metadata, Path(temp_dir))
        
        assert (Path(temp_dir) / "file1_embedding.tif").exists()
        assert (Path(temp_dir) / "file2_embedding.tif").exists()
    gc.collect()


def test_write_batch_temporal_samples(temp_dir):
    """Test writing batch with temporal dimension."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(2, 3, 64, 7, 7)  # B=2, T=3
        file_ids = [["t1_1.tif", "t1_2.tif", "t1_3.tif"], ["t2_1.tif", "t2_2.tif", "t2_3.tif"]]
        metadata = {"time": [["2023-01-01", "2023-01-02", "2023-01-03"], 
                             ["2023-02-01", "2023-02-02", "2023-02-03"]]}
        
        task.write_batch(embedding, file_ids, metadata, Path(temp_dir))
        
        # Check all 6 files were created
        for batch_files in file_ids:
            for file_id in batch_files:
                output_file = Path(temp_dir) / f"{file_id.replace('.tif', '')}_embedding.tif"
                assert output_file.exists()
    gc.collect()


def test_write_batch_parquet_format(temp_dir):
    """Test writing batch in parquet format."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir, output_format="parquet")
        
        embedding = torch.randn(2, 768)
        file_ids = ["file1", "file2"]
        metadata = {}
        
        task.write_batch(embedding, file_ids, metadata, Path(temp_dir))
        
        assert (Path(temp_dir) / "file1_embedding.parquet").exists()
        assert (Path(temp_dir) / "file2_embedding.parquet").exists()
    gc.collect()


# ---- Save Embeddings Tests ----

def test_save_embeddings_tensor(temp_dir):
    """Test saving embeddings from tensor output."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = torch.randn(2, 64, 7, 7)
        file_ids = ["f1.tif", "f2.tif"]
        metadata = {}
        
        task.save_embeddings(embedding, file_ids, metadata, layer=0)
        
        layer_dir = Path(temp_dir) / "layer_0"
        assert layer_dir.exists()
        assert (layer_dir / "f1_embedding.tif").exists()
        assert (layer_dir / "f2_embedding.tif").exists()
    gc.collect()


def test_save_embeddings_dict(temp_dir):
    """Test saving embeddings from dict output (multimodal)."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = {
            'optical': torch.randn(2, 64, 7, 7),
            'radar': torch.randn(2, 32, 7, 7)
        }
        file_ids = ["f1.tif", "f2.tif"]
        metadata = {}
        
        task.save_embeddings(embedding, file_ids, metadata, layer=0)
        
        optical_dir = Path(temp_dir) / "layer_0" / "optical"
        radar_dir = Path(temp_dir) / "layer_0" / "radar"
        assert optical_dir.exists()
        assert radar_dir.exists()
        assert (optical_dir / "f1_embedding.tif").exists()
        assert (radar_dir / "f1_embedding.tif").exists()
    gc.collect()


def test_save_embeddings_invalid_type(temp_dir):
    """Test that invalid embedding type raises TypeError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model", output_dir=temp_dir)
        
        embedding = "invalid"
        file_ids = ["f1.tif"]
        metadata = {}
        
        with pytest.raises(TypeError, match="Unsupported embedding type"):
            task.save_embeddings(embedding, file_ids, metadata, layer=0)
    gc.collect()


# ---- Predict Step Tests ----

def test_predict_step_with_image_dict():
    """Test predict_step when filename is in image dict."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 64, 7, 7)
        mock_registry.build.return_value = model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            task = EmbeddingGenerationTask(model="test_model", output_dir=tmpdir)
            task.configure_models()
            
            batch = {
                'image': {
                    'optical': torch.randn(2, 3, 224, 224),
                    'filename': ["f1.tif", "f2.tif"],
                    'time': ["2023-01-01", "2023-01-02"]
                }
            }
            
            task.predict_step(batch)
            
            # Check files were created
            layer_dir = Path(tmpdir) / "layer_0"
            assert layer_dir.exists()
    gc.collect()


def test_predict_step_with_batch_key():
    """Test predict_step when filename is in batch dict."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 64, 7, 7)
        mock_registry.build.return_value = model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            task = EmbeddingGenerationTask(model="test_model", output_dir=tmpdir)
            task.configure_models()
            
            batch = {
                'image': torch.randn(2, 3, 224, 224),
                'filename': ["f1.tif", "f2.tif"],
                'time': ["2023-01-01", "2023-01-02"]
            }
            
            task.predict_step(batch)
            
            # Check files were created
            layer_dir = Path(tmpdir) / "layer_0"
            assert layer_dir.exists()
    gc.collect()


def test_predict_step_missing_key():
    """Test that missing filename key raises KeyError."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 64, 7, 7)
        mock_registry.build.return_value = model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            task = EmbeddingGenerationTask(model="test_model", output_dir=tmpdir)
            task.configure_models()
            
            batch = {
                'image': torch.randn(2, 3, 224, 224),
                'time': ["2023-01-01", "2023-01-02"]
            }
            
            with pytest.raises(KeyError, match="not found in input dictionary"):
                task.predict_step(batch)
    gc.collect()


def test_predict_step_with_metadata_in_batch():
    """Test predict_step with metadata field in batch."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 64, 7, 7)
        mock_registry.build.return_value = model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            task = EmbeddingGenerationTask(model="test_model", output_dir=tmpdir)
            task.configure_models()
            
            batch = {
                'image': torch.randn(2, 3, 224, 224),
                'filename': ["f1.tif", "f2.tif"],
                'metadata': {
                    'time': ["2023-01-01", "2023-01-02"],
                    'crs': ["EPSG:4326", "EPSG:4326"]
                }
            }
            
            task.predict_step(batch)
            
            layer_dir = Path(tmpdir) / "layer_0"
            assert layer_dir.exists()
    gc.collect()


# ---- Integration Tests ----

def test_full_pipeline_cnn_tiff(temp_dir):
    """Test full pipeline with CNN model and GeoTIFF output."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 64, 7, 7)
        mock_registry.build.return_value = model
        
        task = EmbeddingGenerationTask(
            model="test_cnn",
            output_dir=temp_dir,
            output_format="tiff"
        )
        task.configure_models()
        
        batch = {
            'image': torch.randn(2, 3, 224, 224),
            'filename': ["test1.tif", "test2.tif"]
        }
        
        task.predict_step(batch)
        
        # Verify outputs
        layer_dir = Path(temp_dir) / "layer_0"
        assert (layer_dir / "test1_embedding.tif").exists()
        assert (layer_dir / "test2_embedding.tif").exists()
        
        with rasterio.open(layer_dir / "test1_embedding.tif") as src:
            assert src.count == 64
            assert src.height == 7
            assert src.width == 7
    gc.collect()


def test_full_pipeline_vit_parquet(temp_dir):
    """Test full pipeline with ViT model and GeoParquet output."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(2, 197, 768)
        mock_registry.build.return_value = model
        
        task = EmbeddingGenerationTask(
            model="test_vit",
            output_dir=temp_dir,
            output_format="parquet",
            embedding_pooling="vit_mean",
            has_cls=True
        )
        task.configure_models()
        
        batch = {
            'image': torch.randn(2, 3, 224, 224),
            'filename': ["vit1.tif", "vit2.tif"],
            'time': [np.array("2023-01-01"), np.array("2023-01-02")]
        }
        
        task.predict_step(batch)
        
        # Verify outputs
        layer_dir = Path(temp_dir) / "layer_0"
        output1 = layer_dir / "vit1_embedding.parquet"
        assert output1.exists()
        
        import pandas as pd
        df = pd.read_parquet(output1)
        assert len(df) == 1
        assert "embedding" in df.columns
        assert df["time"][0] == "2023-01-01"
        # After pooling, embedding should be 1D with 768 elements
        assert len(df["embedding"][0]) == 768
    gc.collect()


def test_full_pipeline_multilayer(temp_dir):
    """Test full pipeline with multiple layer outputs."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = [
            torch.randn(2, 32, 14, 14),
            torch.randn(2, 64, 7, 7),
            torch.randn(2, 128, 7, 7)
        ]
        mock_registry.build.return_value = model
        
        task = EmbeddingGenerationTask(
            model="test_multilayer",
            output_dir=temp_dir,
            layers=[0, -1, -2]
        )
        task.configure_models()
        
        batch = {
            'image': torch.randn(2, 3, 224, 224),
            'filename': ["ml1.tif", "ml2.tif"]
        }
        
        task.predict_step(batch)
        
        # Verify all layers saved
        assert (Path(temp_dir) / "layer_0").exists()
        assert (Path(temp_dir) / "layer_1").exists()
        assert (Path(temp_dir) / "layer_2").exists()
    gc.collect()


def test_full_pipeline_multimodal(temp_dir):
    """Test full pipeline with multimodal outputs."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = {
            'optical': torch.randn(2, 64, 7, 7),
            'radar': torch.randn(2, 32, 7, 7)
        }
        mock_registry.build.return_value = model
        
        task = EmbeddingGenerationTask(
            model="test_multimodal",
            output_dir=temp_dir
        )
        task.configure_models()
        
        batch = {
            'image': {
                'optical': torch.randn(2, 3, 224, 224),
                'radar': torch.randn(2, 2, 224, 224)
            },
            'filename': ["mm1.tif", "mm2.tif"]
        }
        
        task.predict_step(batch)
        
        # Verify modality directories created
        optical_dir = Path(temp_dir) / "layer_0" / "optical"
        radar_dir = Path(temp_dir) / "layer_0" / "radar"
        assert optical_dir.exists()
        assert radar_dir.exists()
        assert (optical_dir / "mm1_embedding.tif").exists()
        assert (radar_dir / "mm1_embedding.tif").exists()
    gc.collect()


def test_full_pipeline_temporal(temp_dir):
    """Test full pipeline with temporal data."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        # Return temporal embeddings
        model.return_value = torch.randn(2, 3, 64, 7, 7)  # B=2, T=3
        mock_registry.build.return_value = model
        
        task = EmbeddingGenerationTask(
            model="test_temporal",
            output_dir=temp_dir
        )
        task.configure_models()
        
        batch = {
            'image': torch.randn(2, 6, 3, 224, 224),  # B, C, T, H, W
            'filename': [
                ["t1_1.tif", "t1_2.tif", "t1_3.tif"],
                ["t2_1.tif", "t2_2.tif", "t2_3.tif"]
            ],
            'time': [
                ["2023-01-01", "2023-01-02", "2023-01-03"],
                ["2023-02-01", "2023-02-02", "2023-02-03"]
            ]
        }
        
        task.predict_step(batch)
        
        # Verify all temporal files created
        layer_dir = Path(temp_dir) / "layer_0"
        for i in range(1, 3):
            for j in range(1, 4):
                assert (layer_dir / f"t{i}_{j}_embedding.tif").exists()
    gc.collect()


# ---- Edge Cases and Error Handling ----

@pytest.mark.xfail(reason="Bug in source: IndexError when file_ids is empty")
def test_empty_batch():
    """Test handling of empty batch."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        model = MagicMock()
        model.return_value = torch.randn(0, 64, 7, 7)
        mock_registry.build.return_value = model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            task = EmbeddingGenerationTask(model="test_model", output_dir=tmpdir)
            task.configure_models()
            
            batch = {
                'image': torch.randn(0, 3, 224, 224),
                'filename': [],
                'time': []  # Add empty metadata to match empty batch
            }
            
            # Should handle gracefully
            task.predict_step(batch)
    gc.collect()


def test_configure_callbacks_returns_empty_list():
    """Test that configure_callbacks returns empty list."""
    with patch('terratorch.tasks.embedding_generation.BACKBONE_REGISTRY') as mock_registry:
        mock_registry.build.return_value = MagicMock()
        task = EmbeddingGenerationTask(model="test_model")
        callbacks = task.configure_callbacks()
        assert callbacks == []
    gc.collect()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
