import pygame
import time
import numpy as np
import moderngl

from shaders import UI_VERTEX_SHADERS, UI_FRAGMENT_SHADERS

class SCREEN_OBJ:
    def __init__(self):
        self.WIDTH = 1000
        self.HEIGHT = 800

    def init_screen(self):
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.OPENGL | pygame.DOUBLEBUF
        )

        return self.screen

    def init_clock(self):
        self.clock = pygame.time.Clock()

        return self.clock

class Fps:
    def __init__(self):
        self.frame = 0
        self.frame_push = 0
        self.timer = time.time()

    def update(self):
        self.frame += 1
        if time.time() - self.timer > 1:
            self.frame_push = self.frame
            self.frame = 0
            self.timer = time.time()

class Draw:
    def __init__(self, ctx):
        self.ctx = ctx

        self.fonts = {}

        self.rect_prog = self.ctx.program(vertex_shader=UI_VERTEX_SHADERS['RECT_VS'], fragment_shader=UI_FRAGMENT_SHADERS['RECT_FS'])
        self.text_prog = self.ctx.program(vertex_shader=UI_VERTEX_SHADERS['TEXT_VS'], fragment_shader=UI_FRAGMENT_SHADERS['TEXT_FS'])

        self.rect_vbo = self.ctx.buffer(reserve=6 * 2 * 4)
        self.text_vbo = self.ctx.buffer(reserve=6 * 4 * 4)

        self.rect_vao = self.ctx.vertex_array(self.rect_prog, [(self.rect_vbo, '2f', 'in_position')])
        self.text_vao = self.ctx.vertex_array(self.text_prog, [(self.text_vbo, '2f 2f', 'in_position', 'in_texcoord')])

    def rect(self, position: tuple[int, int], dimensions: tuple[int, int], color: tuple[float, float, float]):
        x, y = position
        w, h = dimensions

        ndc_x = (x / 1000) * 2.0 - 1.0
        ndc_y = 1.0 - (y / 800) * 2.0
        ndc_w = (w / 1000) * 2.0
        ndc_h = (h / 800) * 2.0

        x1 = ndc_x
        y1 = ndc_y
        x2 = ndc_x + ndc_w
        y2 = ndc_y - ndc_h

        vertices = np.array([
            x1, y1,
            x2, y1,
            x2, y2,

            x1, y1,
            x2, y2,
            x1, y2
        ], dtype=np.float32)

        self.rect_vbo.write(vertices.tobytes())

        self.rect_prog['u_color'].value = color
        self.rect_vao.render(moderngl.TRIANGLES)

    def text(self, text_string, font_size=32, text_color=(255, 255, 255), position: tuple[int, int] = (0, 0)):
        font = self.get_font(font_size)
        text_surface = font.render(text_string, True, text_color)

        width, height = text_surface.get_size()

        image_bytes = pygame.image.tobytes(text_surface, 'RGBA', True)

        texture = self.ctx.texture((width, height), 4, image_bytes)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        x, y = position
        x0 = (x / 1000) * 2.0 - 1.0
        y0 = 1.0 - (y / 800) * 2.0
        x1 = ((x + width) / 1000) * 2.0 - 1.0
        y1 = 1.0 - ((y + height) / 800) * 2.0

        quad_data = [
            x0, y0, 0.0, 1.0,
            x0, y1, 0.0, 0.0,
            x1, y1, 1.0, 0.0,

            x0, y0, 0.0, 1.0,
            x1, y1, 1.0, 0.0,
            x1, y0, 1.0, 1.0
        ]

        self.text_vbo.write(bytes(np.array(quad_data, dtype=np.float32)))

        self.ctx.enable(moderngl.BLEND)
        texture.use(0)
        self.text_vao.render()
        self.ctx.disable(moderngl.BLEND)

        texture.release()

    def get_font(self, size):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]