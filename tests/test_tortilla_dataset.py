# Copyright contributors to the Terratorch project
import gc
import pytest

from terratorch.cli_tools import build_lightning_cli

class TestGenericNonGeoSegmentationTortillaDataset:

    @pytest.mark.parametrize("case", ["fit" , "test", "validate"])
    def test_tortilla_file(self, case):
        command_list = [case, "-c", "tests/resources/tortilla/train_segmentation_tortilla.yaml"]
        _ = build_lightning_cli(command_list)

        gc.collect()
        assert True # No exceptions 
    
    def test_missing_tortilla_file(self):
        command_list = ["fit", "-c", "tests/resources/tortilla/train_segmentation_missing.yaml"]
        try:
            _ = build_lightning_cli(command_list)
        except Exception as e:
            assert "Either provide tortilla_file OR all train/val/test roots" in str(e)

        gc.collect()
