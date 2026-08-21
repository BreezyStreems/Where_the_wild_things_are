import threading
import numpy as np

if not hasattr(np, 'product'):
    np.product = np.prod

import random
import queue
import sqlite3
import pickle
import os
import traceback
import pyfastnoisesimd as fns
import math

from skimage.measure import marching_cubes


class World:
    def __init__(self, size=16, seed=None):
        self.size = size
        self.chunk_size = 16
        self.chunk_voxels = {}

        self.requested_chunks = set()
        self.chunk_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.seed = seed if seed is not None else random.randint(0, 2**32-1)

        self.surface_noise = fns.Noise(seed=self.seed, numWorkers=4)
        self.surface_noise.noiseType = fns.NoiseType.Simplex
        self.surface_noise.frequency = 0.008

        self.detail_noise = fns.Noise(seed=self.seed + 1, numWorkers=4)
        self.detail_noise.noiseType = fns.NoiseType.Simplex
        self.detail_noise.frequency = 0.04

        self.cave_noise_gen = fns.Noise(seed=self.seed + 2, numWorkers=4)
        self.cave_noise_gen.noiseType = fns.NoiseType.Simplex
        self.cave_noise_gen.frequency = 0.03

        project_path = os.path.dirname(os.path.realpath(__file__))
        self.db_path = f'{project_path}/World Saves/world_save_{self.seed}.db'
        self._init_db()

        self.worker_thread = threading.Thread(target=self.chunk_worker, daemon=True).start()

    def generate_chunk(self, cx, cy):
        size_xy = self.chunk_size + 1
        size_z = 100

        x_offset = cx * self.chunk_size
        y_offset = cy * self.chunk_size

        z = np.arange(0, size_z, dtype=np.float32)
        Z = z[np.newaxis, np.newaxis, :]

        continental_noise = self.surface_noise.genAsGrid(
            shape=[32, 32, 1], start=[x_offset, y_offset, 0]
        )[:size_xy, :size_xy, 0] * 25.0
        self.cont_noise_for_trans = continental_noise

        detail_noise = self.detail_noise.genAsGrid(
            shape=[32 , 32, 1], start=[x_offset, y_offset, 0]
        )[:size_xy, :size_xy, 0] * 6.0
        self.detail_noise_for_trans = detail_noise

        total_height = 40.0 + continental_noise + detail_noise
        total_height_3d = total_height[:, :, np.newaxis]

        surface_density = total_height_3d - Z

        CAVE_THRESHOLD = 0.07  # Size/frequency of caves
        MIN_CAVE_DEPTH = 10.0  # Depth below surface where caves start fading in
        FADE_DISTANCE = 5.0  # Distance over which caves smoothly reach full size

        cave_noise = self.cave_noise_gen.genAsGrid(
            shape=[32, 32, size_z], start=[x_offset, y_offset, 0]
        )[:size_xy, :size_xy, :size_z]

        caves_cutout = np.abs(cave_noise)

        cave_fade = np.clip((surface_density - MIN_CAVE_DEPTH) / FADE_DISTANCE, 0.0, 1.0)
        cave_carve_amount = np.maximum(0.0, (CAVE_THRESHOLD - caves_cutout)) * 20.0 * cave_fade
        density = surface_density - cave_carve_amount

        materials = np.zeros_like(density, dtype=np.uint8)

        total_height_3d = total_height[:, :, np.newaxis]
        is_surface = (Z >= (total_height_3d - 4)) & (Z <= total_height_3d + 1.5)
        is_underground = (Z < (total_height_3d - 4))
        is_low_elevation = (Z < 32)
        is_cave_air = (density < 0.0) & (surface_density > 0) & (cave_fade > 0)

        materials[is_surface] = 3
        materials[is_underground] = 1
        materials[is_surface & is_low_elevation] = 2
        materials[is_cave_air] = 0

        if np.min(density) > 0.0 or np.max(density) < 0.0:
            return np.array([], dtype=np.float32)

        # --------------------- Higher Detail Mesh ---------------------
        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)
        mat_data[mat_data == 0.0] = 1.0

        v_data = verts[faces.reshape(-1)]
        n_data = normals[faces.reshape(-1)]

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        # --------------------- Lower Detail LOD Mesh ---------------------
        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0, step_size=4)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)
        mat_data[mat_data == 0.0] = 1.0

        v_data = verts[faces.reshape(-1)]
        n_data = normals[faces.reshape(-1)]

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved_LOD = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        water_interleaved, water_interleaved_LOD = self.generate_chunk_transparencies(cx, cy)
        meshes = (interleaved, interleaved_LOD, water_interleaved, water_interleaved_LOD)
        voxels = {'density': density, 'materials': materials}

        return meshes, voxels

    def generate_chunk_transparencies(self, cx, cy):
        size_xy = self.chunk_size + 1

        x_offset = cx * self.chunk_size
        y_offset = cy * self.chunk_size

        z = np.arange(0, 100, dtype=np.float32)
        Z = z[np.newaxis, np.newaxis, :]

        total_height = 40.0 + self.cont_noise_for_trans + self.detail_noise_for_trans
        total_height_3d = total_height[:, :, np.newaxis]

        water_level = 32.0
        top_surface = water_level - Z
        bottom_surface = Z - total_height_3d
        density = np.minimum(top_surface, bottom_surface)

        materials = np.ones_like(density, dtype=np.uint8) * 4

        surface_z = np.clip(np.floor(total_height).astype(int), 0, 99)
        x_idx, y_idx = np.ogrid[:size_xy, :size_xy]
        surface_material = materials[x_idx, y_idx, surface_z]

        is_not_sand = (surface_material != 2)

        density[is_not_sand, :] = -1.0

        if np.min(density) > 0.0 or np.max(density) < 0.0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        #---------------------Higher Detail---------------

        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)

        v_data = verts[faces.reshape(-1)]
        n_data = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(v_data), 1))

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        #-------------------------Lower Detail------------------

        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0, step_size=4)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)

        v_data = verts[faces.reshape(-1)]
        n_data = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(v_data), 1))

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved_LOD = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        return interleaved, interleaved_LOD

    def chunk_worker(self):
        db_conn = sqlite3.connect(self.db_path)
        cursor = db_conn.cursor()

        while True:
            cx, cy = self.chunk_queue.get()
            try:
                cursor.execute("SELECT data FROM chunks WHERE cx=? AND cy=?", (cx, cy))
                row = cursor.fetchone()

                if row:
                    meshes, voxels = pickle.loads(row[0])
                else:
                    meshes, voxels = self.generate_chunk(cx, cy)
                    cursor.execute("INSERT INTO chunks (cx, cy, data) VALUES (?, ?, ?)",
                                   (cx, cy, pickle.dumps((meshes, voxels))))
                    db_conn.commit()

                self.result_queue.put(((cx, cy), meshes, voxels))
            except:
                traceback.print_exc()
            finally:
                self.chunk_queue.task_done()

    def request_chunk(self, cx, cy):
        if (cx, cy) not in self.requested_chunks:
            self.requested_chunks.add((cx, cy))
            self.chunk_queue.put((cx, cy))

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS chunks 
            (cx INTEGER, cy INTEGER, data BLOB, PRIMARY KEY (cx, cy))''')

    def dig(self, radius, mx, my, mz):
        mx, my, mz = math.floor(mx), math.floor(my), math.floor(mz)
        print(f'dug: {mx, my, mz}')
        chunks_to_refresh = []
        for i in range(mx - radius, mx + radius):
            for o in range(my - radius, my + radius):
                for p in range(mz - radius, mz + radius):
                    distance = math.sqrt((i - mx)**2 + (o - my)**2 + (p - mz)**2)

                    if distance <= radius:
                        cx = math.floor(i/ self.chunk_size)
                        cy = math.floor(o / self.chunk_size)

                        chunks_to_refresh.append((cx, cy))

                        lx = int(math.floor(i)) % self.chunk_size
                        ly = int(math.floor(o)) % self.chunk_size
                        lz = int(math.floor(p))

                        voxels = self.chunk_voxels[(cx, cy)]
                        voxels['materials'][lx, ly, lz] = 0
                        voxels['density'][lx, ly, lz] = 0
                        self.chunk_voxels[(cx, cy)] = voxels

        chunks_to_refresh = list(dict.fromkeys(chunks_to_refresh))

        db_conn = sqlite3.connect(self.db_path)
        cursor = db_conn.cursor()

        for chunk in chunks_to_refresh:
            cx, cy = chunk

            cursor.execute("SELECT data FROM chunks WHERE cx=? AND cy=?", (cx, cy))
            meshes, voxels = self.generate_chunk_mesh(cx, cy)
            data = pickle.dumps((meshes, voxels))
            cursor.execute(
                "UPDATE chunks SET data = ? WHERE cx = ? AND cy = ?",
                (data, cx, cy)
            )
            db_conn.commit()

    def get_picked_block(self, screen_w, screen_h, view_matrix, proj_matrix, camera_pos):
        ndc_x = (2.0 * (screen_w / 2)) / screen_w - 1.0
        ndc_y = 1.0 - (2.0 * (screen_h / 2)) / screen_h

        ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)

        inv_proj = np.linalg.inv(proj_matrix)
        ray_eye = inv_proj @ ray_clip
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], dtype=np.float32)

        inv_view = np.linalg.inv(view_matrix)
        ray_world = (inv_view @ ray_eye)[:3]

        ray_dir = ray_world / np.linalg.norm(ray_world)

        current_dist = 0.0
        max_dist = 10
        ray_origin = np.array(camera_pos, dtype=np.float32)
        step_size = 0.2


        while current_dist < max_dist:
            pos = ray_origin + ray_dir * current_dist

            world_x = pos[0]
            world_y = pos[1]
            world_z = pos[2]

            cx = int(np.floor(world_x / self.chunk_size))
            cy = int(np.floor(world_y / self.chunk_size))

            local_x = int(np.floor(world_x - cx * self.chunk_size))
            local_y = int(np.floor(world_y - cy * self.chunk_size))
            local_z = int(np.floor(world_z))

            if 0 <= local_x < self.chunk_size and 0 <= local_y < self.chunk_size and 0 <= local_z < 100:
                mat_id = self.get_voxel_at(local_x, local_y, local_z)
                mat_id = mat_id[1]

                if mat_id in (1, 2, 3):
                    return {
                        "chunk": (cx, cy),
                        "local_voxel": (local_x, local_y, local_z),
                        "hit_pos": pos,
                        "material": mat_id
                    }

            current_dist += step_size

    def get_voxel_at(self, world_x, world_y, world_z):
        if not (0 <= world_z < 100):
            return None, None

        cx = math.floor(world_x / self.chunk_size)
        cy = math.floor(world_y / self.chunk_size)

        lx = int(math.floor(world_x)) % self.chunk_size
        ly = int(math.floor(world_y)) % self.chunk_size
        lz = int(math.floor(world_z))

        if (cx, cy) in self.chunk_voxels:
            voxels = self.chunk_voxels[(cx, cy)]
            return voxels['density'][lx, ly, lz], voxels['materials'][lx, ly, lz]

        return None, None

    def is_solid(self, world_x, world_y, world_z):
        density, _ = self.get_voxel_at(world_x, world_y, world_z)
        return density is not None and density > 0.0

    def generate_chunk_mesh(self, cx, cy):
        size_xy = self.chunk_size + 1
        size_z = 100

        x_offset = cx * self.chunk_size
        y_offset = cy * self.chunk_size

        voxels = self.chunk_voxels[(cx, cy)]
        density = voxels['density']
        materials = voxels['materials']

        if np.min(density) > 0.0 or np.max(density) < 0.0:
            return np.array([], dtype=np.float32)

        # --------------------- Higher Detail Mesh ---------------------
        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)
        mat_data[mat_data == 0.0] = 1.0

        v_data = verts[faces.reshape(-1)]
        n_data = normals[faces.reshape(-1)]

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        # --------------------- Lower Detail LOD Mesh ---------------------
        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0, step_size=4)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)
        mat_data[mat_data == 0.0] = 1.0

        v_data = verts[faces.reshape(-1)]
        n_data = normals[faces.reshape(-1)]

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved_LOD = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        water_interleaved, water_interleaved_LOD = self.generate_chunk_trans_meshes(cx, cy)
        meshes = (interleaved, interleaved_LOD, water_interleaved, water_interleaved_LOD)
        voxels = {'density': density, 'materials': materials}

        return meshes, voxels

    def generate_chunk_trans_meshes(self, cx, cy):
        size_xy = self.chunk_size + 1

        x_offset = cx * self.chunk_size
        y_offset = cy * self.chunk_size

        voxels = self.chunk_voxels[(cx, cy)]
        density = voxels['density']
        materials = voxels['materials']

        if np.min(density) > 0.0 or np.max(density) < 0.0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # ---------------------Higher Detail---------------

        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)

        v_data = verts[faces.reshape(-1)]
        n_data = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(v_data), 1))

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        # -------------------------Lower Detail------------------

        try:
            verts, faces, normals, _ = marching_cubes(density, level=0.0, step_size=4)
        except (ValueError, RuntimeError):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        verts[:, 0] += x_offset
        verts[:, 1] += y_offset

        local_x = np.clip(verts[:, 0] - x_offset, 0, self.chunk_size).astype(int)
        local_y = np.clip(verts[:, 1] - y_offset, 0, self.chunk_size).astype(int)
        local_z = np.clip(verts[:, 2], 0, 100 - 1).astype(int)

        mat_data = materials[local_x, local_y, local_z].astype(np.float32).reshape(-1, 1)

        v_data = verts[faces.reshape(-1)]
        n_data = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(v_data), 1))

        mat_data_interleaved = mat_data[faces.reshape(-1)]
        interleaved_LOD = np.hstack((v_data, n_data, mat_data_interleaved)).astype(np.float32)

        return interleaved, interleaved_LOD