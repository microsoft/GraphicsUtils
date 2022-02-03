# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import zipfile
from io import BytesIO
from typing import List
from abc import abstractmethod

import cv2
import numpy as np


class ZipBase(object):
    """
    The base class of a zip instance
    """
    def __init__(self):
        pass

    @abstractmethod
    def namelist(self):
        """
        Get the files list
        """

    def listdir(self, dir_name):
        """
        List all files of a given sub folder
        """
        f_asb = lambda _s: _s[_s.find(dir_name) + len(dir_name):]
        file_list = [f_asb[_f] for _f in self.namelist() if _f.find(dir_name) != -1]
        file_list = [_f.lstrip('/').rstrip('/') for _f in file_list]
        file_list = [_f for _f in file_list if len(_f.split('/')) == 1 and _f.split('/')[0]]
        return file_list

    @abstractmethod
    def has_file(self, file_name):
        """
        Whether a file in the zip
        """

    @abstractmethod
    def read(self, file_name):
        """
        Read the binary string
        """

    @abstractmethod
    def read_image(self, file_name, rgb=True):
        """
        Read image from zip
        """

    @abstractmethod
    def close(self):
        """
        Close the zip stream
        """


class ZipIO(ZipBase):
    """
    Zip operation on a single zip file
    """
    def __init__(self, zip_path, mode='r', prefetch=False):
        super().__init__()
        if prefetch:
            raise NotImplementedError
        else:
            self._zip_meta = zipfile.ZipFile(zip_path, mode)

    def namelist(self) -> List[str]:
        return sorted(self._zip_meta.namelist())

    def has_file(self, file_name):
        return file_name in self._zip_meta.namelist()

    def read(self, file_name):
        if isinstance(file_name, list):
            file_name = '/'.join(tuple(file_name))
        return self._zip_meta.read(file_name)

    def writestr(self, file_name, data):
        """
        Write the binary string
        """
        return self._zip_meta.writestr(file_name, data)

    def read_image(self, file_name, rgb=True):
        sample_meta = self._zip_meta.read(file_name)
        img = cv2.imdecode(np.frombuffer(sample_meta, np.uint8), cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
        if rgb and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.array(img)

    def read_npy(self, file_name):
        """
        Read NPY file from zip
        """
        bytes_stream = BytesIO(self._zip_meta.read(file_name))
        return np.load(bytes_stream)

    def close(self):
        self._zip_meta.close()


class GroupZipIO(ZipBase):
    """
    Zip operation on a folder containing many zips
    """
    def __init__(self, zip_dir, mode='r', prefetch=False):
        super().__init__()
        if isinstance(zip_dir, list):
            zip_files = zip_dir
        elif os.path.isdir(zip_dir):
            zip_files = [os.path.join(zip_dir, f) for f in os.listdir(zip_dir)]
        else:
            raise AttributeError
        self._zips_meta: List[zipfile.ZipFile] = [zipfile.ZipFile(f, mode) for f in zip_files]
        if mode != 'r':
            raise NotImplementedError
        if prefetch:
            raise NotImplementedError
        self._names_list = list()
        self._names_map = dict()
        for z_idx, z_m in enumerate(self._zips_meta):
            self._names_list.extend(z_m.namelist())
            for _n in z_m.namelist():
                self._names_map[_n] = z_idx

    def namelist(self):
        return self._names_list

    def has_file(self, file_name):
        return file_name in self._names_list

    def read(self, file_name):
        if isinstance(file_name, list):
            file_name = '/'.join(tuple(file_name))
        z_idx = self._names_map[file_name]
        return self._zips_meta[z_idx].read(file_name)

    def read_image(self, file_name, rgb=True):
        sample_meta = self.read(file_name)
        img = cv2.imdecode(np.frombuffer(sample_meta, np.uint8), cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member
        if rgb and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.array(img)

    def close(self):
        for z_m in self._zips_meta:
            z_m.close()
