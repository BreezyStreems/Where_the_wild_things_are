import pygame
import math
import numpy as np

from pyglm import glm

class Camera:
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.Z = 120

        self.chunk_row = 0
        self.chunk_col = 0

        self.pitch = -15
        self.yaw = 90

        self.fov = 60.0
        self.render_dist = 5
        self.eye_height = 2.0

    def get_view_matrix(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        front = glm.vec3(
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        )
        front = glm.normalize(front)

        pos = glm.vec3(self.X, self.Y, self.Z)
        up = glm.vec3(0.0, 0.0, 1.0)

        return glm.lookAt(pos, pos + front, up)

    def physics_calc(self, speed, moving, world, direction=None, key=None):
        gx = math.floor(self.X)
        gy = math.floor(self.Y)
        gz_feet = int(math.floor(self.Z - self.eye_height))

        if not world.is_solid(gx, gy, gz_feet - 1):
            self.Z -= speed
            pass
        else:
            if moving:
                headroom_clear = not world.is_solid(gx, gy, gz_feet + 2) and not world.is_solid(gx, gy, gz_feet + 3)
                if headroom_clear:
                    self.Z += 1
                else:
                    if key[pygame.K_w]:
                        self.X -= direction[0] * speed
                        self.Y -= direction[1] * speed
                    if key[pygame.K_s]:
                        self.X += direction[0] * speed
                        self.Y += direction[1] * speed
                    if key[pygame.K_d]:
                        self.X -= direction[2] * speed
                        self.Y -= direction[3] * speed
                    if key[pygame.K_a]:
                        self.X += direction[2] * speed
                        self.Y += direction[3] * speed

    def get_chunk(self):
        self.chunk_row = int(math.floor(self.X / 16.0))
        self.chunk_col = int(math.floor(self.Y / 16.0))

    def projection_matrix(self, aspect_ratio, near=0.1, far=1000.0):
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0

        return proj