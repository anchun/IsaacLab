# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import hashlib
import json
import os
import pathlib
import random
from datetime import datetime

from omni.isaac.orbit.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from omni.isaac.orbit.utils.assets import check_file_path
from omni.isaac.orbit.utils.io import dump_yaml


class AssetConverterBase(abc.ABC):
    """Base class for converting an asset file from different formats into USD format.

    This class provides a common interface for converting an asset file into USD. It does not
    provide any implementation for the conversion. The derived classes must implement the
    :meth:`_convert_asset` method to provide the actual conversion.

    The file conversion is lazy if the output directory (:obj:`AssetConverterBaseCfg.usd_dir`) is provided.
    In the lazy conversion, the USD file is re-generated only if:

    * The asset file is modified.
    * The configuration parameters are modified.
    * The USD file does not exist.

    To override this behavior to force conversion, the flag :obj:`AssetConverterBaseCfg.force_usd_conversion`
    can be set to True.

    When no output directory is defined, lazy conversion is deactivated and the generated USD file is
    stored in folder ``/tmp/Orbit/usd_{date}_{time}_{random}``, where the parameters in braces are generated
    at runtime. The random identifiers help avoid a race condition where two simultaneously triggered conversions
    try to use the same directory for reading/writing the generated files.

    .. note::
        Changes to the parameters :obj:`AssetConverterBaseCfg.asset_path`, :obj:`AssetConverterBaseCfg.usd_dir`, and
        :obj:`AssetConverterBaseCfg.usd_file_name` are not considered as modifications in the configuration instance that
        trigger USD file re-generation.

    """

    def __init__(self, cfg: AssetConverterBaseCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for converting an asset file to USD format.

        Raises:
            ValueError: When provided asset file does not exist.
        """
        # check if the asset file exists
        if not check_file_path(cfg.asset_path):
            raise ValueError(f"The asset path does not exist: {cfg.asset_path}")
        # save the inputs
        self.cfg = cfg

        # resolve USD directory name
        if cfg.usd_dir is None:
            # a folder in "/tmp/Orbit" by the name: usd_{date}_{time}_{random}
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._usd_dir = f"/tmp/Orbit/usd_{time_tag}_{random.randrange(10000)}"
        else:
            self._usd_dir = cfg.usd_dir

        # resolve the file name from asset file name if not provided
        if cfg.usd_file_name is None:
            usd_file_name = pathlib.PurePath(cfg.asset_path).stem
        else:
            usd_file_name = cfg.usd_file_name
        # add USD extension if not provided
        if not (usd_file_name.endswith(".usd") or usd_file_name.endswith(".usda")):
            self._usd_file_name = usd_file_name + ".usda"
        else:
            self._usd_file_name = usd_file_name

        # create the USD directory
        os.makedirs(self.usd_dir, exist_ok=True)
        # check if usd files exist
        self._usd_file_exists = os.path.isfile(self.usd_path)
        # path to read/write asset hash file
        self._dest_hash_path = os.path.join(self.usd_dir, ".asset_hash")
        # create asset hash to check if the asset has changed
        self._asset_hash = self._config_to_hash(cfg)
        # read the saved hash
        try:
            with open(self._dest_hash_path) as f:
                existing_asset_hash = f.readline()
                self._is_same_asset = existing_asset_hash == self._asset_hash
        except FileNotFoundError:
            self._is_same_asset = False

        # convert the asset to USD if the hash is different or USD file does not exist
        if cfg.force_usd_conversion or not self._usd_file_exists or not self._is_same_asset:
            # write the updated hash
            with open(self._dest_hash_path, "w") as f:
                f.write(self._asset_hash)
            # convert the asset to USD
            self._convert_asset(cfg)
            # dump the configuration to a file
            dump_yaml(os.path.join(self.usd_dir, "config.yaml"), cfg.to_dict())

    """
    Properties.
    """

    @property
    def usd_dir(self) -> str:
        """The absolute path to the directory where the generated USD files are stored."""
        return self._usd_dir

    @property
    def usd_file_name(self) -> str:
        """The file name of the generated USD file."""
        return self._usd_file_name

    @property
    def usd_path(self) -> str:
        """The absolute path to the generated USD file."""
        return os.path.join(self.usd_dir, self.usd_file_name)

    @property
    def usd_instanceable_meshes_path(self) -> str:
        """The relative path to the USD file with meshes.

        The path is with respect to the USD directory :attr:`usd_dir`. This is to ensure that the
        mesh references in the generated USD file are resolved relatively. Otherwise, it becomes
        difficult to move the USD asset to a different location.
        """
        return os.path.join(".", "Props", "instanceable_meshes.usda")

    """
    Implementation specifics.
    """

    @abc.abstractmethod
    def _convert_asset(self, cfg: AssetConverterBaseCfg):
        """Converts the asset file to USD.

        Args:
            cfg: The configuration instance for the input asset to USD conversion.
        """
        raise NotImplementedError()

    """
    Private helpers.
    """

    @staticmethod
    def _config_to_hash(cfg: AssetConverterBaseCfg) -> str:
        """Converts the configuration object and asset file to an MD5 hash of a string.

        .. warning::
            It only checks the main asset file (:attr:`cfg.asset_path`).

        Args:
            config : The asset converter configuration object.

        Returns:
            An MD5 hash of a string.
        """

        # convert to dict and remove path related info
        config_dic = cfg.to_dict()
        _ = config_dic.pop("asset_path")
        _ = config_dic.pop("usd_dir")
        _ = config_dic.pop("usd_file_name")
        # convert config dic to bytes
        config_bytes = json.dumps(config_dic).encode()
        # hash config
        md5 = hashlib.md5()
        md5.update(config_bytes)

        # read the asset file to observe changes
        with open(cfg.asset_path, "rb") as f:
            while True:
                # read 64kb chunks to avoid memory issues for the large files!
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        # return the hash
        return md5.hexdigest()
