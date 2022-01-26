# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import cv2
import zipfile
import logging
import numpy as np

from typing import List, Optional
from abc import abstractmethod
from typing import List, Optional
from plyfile import PlyData, PlyElement

from .miscellaneous import palettes


class PlyIO(object):
    def __init__(self):
        self.vertices: Optional[np.ndarray] = None
        self.vertices_color: Optional[np.ndarray] = None
        self.lines: Optional[np.ndarray] = None
        self.lines_color: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.faces_color: Optional[np.ndarray] = None

    def add_vertices(self, vertices: np.ndarray, colors: np.ndarray = None):
        vertices = np.reshape(vertices, [-1, vertices.shape[-1]]).astype(np.float32)
        colors = np.reshape(colors, [-1, colors.shape[-1]]) if colors is not None else None
        assert vertices.shape[-1] in [3, 6]
        if vertices.shape[-1] == 6:
            colors = vertices[:, 3:]
            vertices = vertices[:, :3]
        elif colors is None:
            colors = np.zeros(vertices.shape) + 255
        colors = np.array(colors).astype(np.uint8)
        assert vertices.shape[-1] == 3
        assert colors.shape[-1] == 3
        self.vertices = vertices if self.vertices is None else np.concatenate([self.vertices, vertices], axis=0)
        self.vertices_color = colors if self.vertices_color is None else np.concatenate([self.vertices_color, colors],
                                                                                        axis=0)

    def get_num_vertices(self):
        return 0 if self.vertices is None else self.vertices.shape[0]

    def add_faces(self, vertices, faces, vertices_color=None, faces_color=None):
        face_vertex_start = self.get_num_vertices()
        self.add_vertices(vertices, vertices_color)

        faces = np.reshape(faces, [-1, faces.shape[-1]]).astype(np.int32)
        assert faces.shape[-1] in [3, 4, 6, 7]
        if faces.shape[-1] == 6:
            faces_color = faces[:, 3:]
            faces = faces[:, :3]
        elif faces.shape[-1] == 7:
            faces_color = faces[:, 4:]
            faces = faces[:, :4]
        assert faces.shape[-1] in [3, 4]
        self.faces = faces if self.faces is None else np.concatenate([self.faces, faces + face_vertex_start], axis=0)

        if faces_color is not None:
            assert faces_color.shape[-1] == 3
            face_color = np.reshape(faces_color, [-1, faces_color.shape[-1]]).astype(np.uint8)
            self.faces_color = face_color if self.faces_color is None else np.concatenate(
                [self.faces_color, face_color], axis=0)

    def add_lines(self, vertices, lines, vertices_color=None, lines_color=None):
        line_vertex_start = 0 if self.vertices is None else self.vertices.shape[0]
        self.add_vertices(vertices, vertices_color)

        lines = np.reshape(lines, [-1, lines.shape[-1]]).astype(np.int32)
        assert lines.shape[-1] in [2, 5]
        if lines.shape[-1] == 5:
            lines_color = lines[:, 2:]
            lines = lines[:, :2]
        assert lines.shape[-1] == 2
        self.lines = lines if self.lines is None else np.concatenate([self.lines, lines + line_vertex_start], axis=0)

        if lines_color is not None:
            assert lines_color.shape[-1] == 3
            line_color = np.reshape(lines_color, [-1, lines_color.shape[-1]]).astype(np.uint8)
            self.lines_color = line_color if self.lines_color is None else np.concatenate(
                [self.lines_color, line_color], axis=0)

    def add_box(self, box_vertices, colors=None):
        box_vertices = np.reshape(box_vertices, [-1, box_vertices.shape[-1]]).astype(np.float32)
        assert box_vertices.shape[-1] in [3, 6]
        if box_vertices.shape[-1] == 6:
            colors = box_vertices[:, 3:]
            box_vertices = box_vertices[:, :3]
        elif colors is None:
            colors = np.zeros(box_vertices.shape, dtype=np.uint8) + 255

        edge_np = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                            [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int32)
        face_np = np.array([[3, 2, 1], [3, 1, 0], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                            [7, 6, 2], [7, 2, 3], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], dtype=np.int32)
        obb_edge_list = np.concatenate([edge_np + v_i for v_i in range(0, len(box_vertices), 8)], axis=0)
        obb_face_list = np.concatenate([face_np + v_i for v_i in range(0, len(box_vertices), 8)], axis=0)
        face_rgbs_list = np.concatenate([[colors[v_i]] * 12 for v_i in range(0, len(box_vertices), 8)], axis=0)
        edge_colors_list = np.concatenate([[colors[v_i]] * 12 for v_i in range(0, len(box_vertices), 8)], axis=0)

        self.add_faces(box_vertices, obb_face_list, vertices_color=colors, faces_color=face_rgbs_list)
        self.add_lines(box_vertices, obb_edge_list, vertices_color=colors, lines_color=edge_colors_list)

    def load(self, file_path):
        pass

    def dump(self, file_path):
        ply_elem = list()
        ply_vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        out_vertices = self.vertices
        if self.vertices_color is not None:
            out_vertices = np.concatenate([self.vertices, self.vertices_color], axis=-1)
            ply_vertex_type.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        out_vertices_tuple = np.array([tuple(r) for r in out_vertices], dtype=ply_vertex_type)
        vertex_elem = PlyElement.describe(out_vertices_tuple, 'vertex')
        ply_elem.append(vertex_elem)

        if self.lines is not None:
            ply_line_type = [('vertex1', 'i4'), ('vertex2', 'i4')]
            out_lines = self.lines
            if self.lines_color is not None:
                out_lines = np.concatenate([self.lines, self.lines_color], axis=-1)
                ply_line_type.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            out_lines_tuple = np.array([tuple(r) for r in out_lines], dtype=ply_line_type)
            line_elem = PlyElement.describe(out_lines_tuple, 'edge')
            ply_elem.append(line_elem)

        if self.faces is not None:
            face_dim = self.faces.shape[-1]
            ply_face_type = [('vertex_indices', 'i4', (face_dim,))]
            out_faces = self.faces
            if self.faces_color is not None:
                out_faces = np.concatenate([self.faces, self.faces_color], axis=-1)
                ply_face_type.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                out_faces_tuple = np.array([(tuple(r[:face_dim]), *tuple(r[face_dim:])) for r in out_faces],
                                           dtype=ply_face_type)
            else:
                out_faces_tuple = np.array([(tuple(r[:face_dim]),) for r in out_faces], dtype=ply_face_type)
            face_elem = PlyElement.describe(out_faces_tuple, 'face')
            ply_elem.append(face_elem)

        export_ply_data = PlyData(ply_elem, text=False, byte_order='<')
        export_ply_data.write(file_path)

    @staticmethod
    def point_vis_as_box(vertices, vertices_color, box_scale):
        point_offset = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1],
                                 [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]]) * box_scale
        vertices_tile = np.tile(np.expand_dims(vertices, axis=1), (1, 8, 1))  # n 3 -> n 8 3
        box_vertices = vertices_tile + point_offset
        box_vertices = np.asarray(box_vertices).reshape((-1, 3))

        box_colors = np.tile(np.expand_dims(vertices_color, axis=1), (1, 8, 1)).reshape((-1, 3))

        box_face = np.array([[3, 1, 0], [3, 2, 1], [4, 5, 7], [5, 6, 7],
                             [0, 1, 4], [5, 4, 1], [2, 3, 7], [7, 6, 2],
                             [1, 2, 5], [6, 5, 2], [4, 3, 0], [3, 4, 7]])
        box_faces = [box_face + f_i * 8 for f_i in range(len(vertices))]
        box_faces = np.asarray(box_faces).reshape((-1, 3))
        box_face_colors = np.tile(np.expand_dims(vertices_color, axis=1), (1, 12, 1)).reshape((-1, 3))

        return box_vertices, box_faces, box_colors, box_face_colors

    def dump_point_as_box(self, file_path, vox_scale=1):
        assert self.lines is None and self.faces is None
        box_vertices, box_faces, box_colors, face_c = self.point_vis_as_box(self.vertices, self.vertices_color,
                                                                            vox_scale / 2)
        self.vertices, self.vertices_color = None, None
        self.add_faces(box_vertices, box_faces, box_colors, face_c)
        self.dump(file_path)

    def dump_vox(self, file_path, vox_data, vis_as_box=True, vox_scale=1, colors_map=palettes.d3c20_rgb()):
        vox_data = np.asarray(vox_data)
        if np.any(vox_data >= len(colors_map)):
            logging.warning(f'Label number is more than color number!!!')
            vox_data = np.asarray(vox_data) % len(colors_map)
        vox_indices = np.argwhere(vox_data > 0) * vox_scale
        vox_label = vox_data[vox_data > 0]
        vox_colors = np.asarray(colors_map)[vox_label]
        self.add_vertices(vox_indices, vox_colors)
        if vis_as_box:
            self.dump_point_as_box(file_path, vox_scale)
        else:
            self.dump(file_path)


class MeshObjIO(object):
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.face_vertices = None

    def load(self, file_path):
        self.vertices = list()
        self.faces = list()
        self.face_vertices = list()
        with open(file_path, 'rb') as fp:
            _ = fp.readline()
            while True:
                c = fp.readline().rstrip().decode('utf-8').split(' ')
                attr_name = c[0]
                attr_values = c[1:]
                if not attr_name:
                    break
                elif attr_name == 'v':
                    self.vertices.append([float(a) for a in attr_values])
                elif attr_name == 'f':
                    self.faces.append([int(a.split('/')[0]) for a in attr_values])
            self.vertices = np.array(self.vertices, dtype=np.float32)
            self.faces = np.array(self.faces, dtype=np.int32) - 1
            self.face_vertices = self.vertices[self.faces, :]


class ZipBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def namelist(self): pass

    def listdir(self, dir_name):
        file_list = [f[f.find(dir_name) + len(dir_name):] for f in self.namelist() if f.find(dir_name) != -1]
        file_list = [f.lstrip('/').rstrip('/') for f in file_list]
        file_list = [f for f in file_list if len(f.split('/')) == 1 and f.split('/')[0]]
        return file_list


class ZipIO(ZipBase):
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
        return self._zip_meta.writestr(file_name, data)

    def read_image(self, file_name, rgb=True):
        sample_meta = self._zip_meta.read(file_name)
        img = cv2.imdecode(np.frombuffer(sample_meta, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.array(img)

    def close(self):
        self._zip_meta.close()


class GroupZipIO(ZipBase):
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
            for n in z_m.namelist():
                self._names_map[n] = z_idx

    def namelist(self):
        return self._names_list

    def has_file(self, file_name):
        return file_name in self._names_list

    def get_file_list(self):
        raise NotImplementedError

    def read(self, file_name):
        if isinstance(file_name, list):
            file_name = '/'.join(tuple(file_name))
        z_idx = self._names_map[file_name]
        return self._zips_meta[z_idx].read(file_name)

    def read_image(self, file_name, rgb=True):
        sample_meta = self.read(file_name)
        img = cv2.imdecode(np.frombuffer(sample_meta, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb and img.shape[-1] == 3:
            img = img[..., ::-1]
        return np.array(img)

    def close(self):
        for z_m in self._zips_meta:
            z_m.close()


class TextIO(object):
    def __init__(self, text_path, mode='r'):
        # self._text_meta = open(text_path, mode=mode)
        self._text_path = text_path
        self._mode = mode
        self._text_meta = open(text_path, mode)

    def read_lines(self):
        return [f.rstrip() for f in self._text_meta.readlines()]

    def close(self):
        self._text_meta.close()


class OpenMeta(object):
    def __init__(self, path, mode):
        self._path = path
        self._mode = mode
        self._meta = None

    @abstractmethod
    def _open(self): pass

    def __enter__(self):
        self._open()
        return self._meta

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._meta.close()


class OpenText(OpenMeta):
    def __init__(self, path, mode):
        super().__init__(path, mode)

    def __enter__(self):
        return super().__enter__()

    def _open(self):
        self._meta = TextIO(self._path, self._mode)
