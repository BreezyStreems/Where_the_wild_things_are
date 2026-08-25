import pygame
import sys
import math
import moderngl
from mouseinfo import position

from pyglm import glm
from typing import Any
from utils import Fps, SCREEN_OBJ, Draw
from world import World
from camera import Camera
from shaders import FRAGMENT_SHADER, VERTEX_SHADER

pygame.init()

screen = SCREEN_OBJ().init_screen()
pygame.display.set_caption("horror game engine")
pygame.event.set_grab(False)
pygame.mouse.set_visible(True)
pygame.mouse.get_rel()

# -------------WORLD CREATION VARIABLES------------------
seed = 1000000

ctx = moderngl.create_context()
ctx.enable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

camera = Camera()
world = World(size=camera.render_dist * 16, seed=seed)
fps = Fps()
draw = Draw(ctx)

proj = glm.perspective(glm.radians(camera.fov), 1000.0 / 800.0, 0.1, 1000.0)
prog['u_proj'].write(proj.to_bytes())

chunk_vaos = {}
clock = SCREEN_OBJ().init_clock()
running = True
sensitivity = 0.15

state = 'menu'
chunk_count = 0

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)

    if state == 'menu':
        ctx.clear(0.8, 0.8, 1)
        draw.rect((400, 350), (200, 100), (0, 0, 0))

        mouse_pos = pygame.mouse.get_pos()
        mouse_state = pygame.mouse.get_pressed()

        if mouse_state[0]:
            if 400 <= mouse_pos[0] <= 600 and 350 <= mouse_pos[1] <= 450:
                state = 'loading world'
                camera.get_chunk()

    if state == 'loading world':
        ctx.clear(0.8, 0.8, 1)
        if not chunk_count == 0:
            for i in range(chunk_count):
                x = (i / (((camera.render_dist + 10) * 2)**2)) * 100 * 10
                draw.rect((x, 100), (1, 100), (0, 0, 0))

        for cx in range(camera.chunk_row - (camera.render_dist + 10), camera.chunk_row + camera.render_dist + 10):
            for cy in range(camera.chunk_col - (camera.render_dist + 10), camera.chunk_col + camera.render_dist + 10):
                if (cx, cy) not in chunk_vaos:
                    world.request_chunk(cx, cy)

        chunks_uploaded_this_frame = 0
        while not world.result_queue.empty():
            (cx, cy), meshes, voxels = world.result_queue.get()
            mesh_data, low_res_mesh_data, transparencies_mesh_data, low_res_transparencies_mesh_data = meshes
            world.chunk_voxels[(cx, cy)] = voxels

            chunk_entry: dict[str, Any] = {
                'count': len(mesh_data),
                'low_detail_count': len(low_res_mesh_data),
                'trans_count': len(transparencies_mesh_data),
                'trans_lod_count': len(low_res_transparencies_mesh_data)
            }

            if len(mesh_data) > 0:
                vbo = ctx.buffer(mesh_data.tobytes())
                chunk_entry['vbo'] = vbo
                chunk_entry['vao'] = ctx.vertex_array(prog,
                                                      [(vbo, '3f 3f 1f', 'in_position', 'in_normal', 'in_material')])

            if len(low_res_mesh_data) > 0:
                low_vbo = ctx.buffer(low_res_mesh_data.tobytes())
                chunk_entry['low_detail_vbo'] = low_vbo
                chunk_entry['low_detail_vao'] = ctx.vertex_array(prog,
                                                                 [(low_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                   'in_material')])

            if len(transparencies_mesh_data) > 0:
                trans_vbo = ctx.buffer(transparencies_mesh_data.tobytes())
                chunk_entry['trans_vbo'] = trans_vbo
                chunk_entry['trans_vao'] = ctx.vertex_array(prog, [(trans_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                    'in_material')])

            if len(low_res_transparencies_mesh_data) > 0:
                trans_lod_vbo = ctx.buffer(low_res_transparencies_mesh_data.tobytes())
                chunk_entry['trans_lod_vbo'] = trans_lod_vbo
                chunk_entry['trans_lod_vao'] = ctx.vertex_array(prog,
                                                                [(trans_lod_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                  'in_material')])

            chunk_vaos[(cx, cy)] = chunk_entry

            chunk_count += 1

        if chunk_count == ((camera.render_dist + 10) * 2) ** 2:
            state = 'game'
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)

    if state == 'game':
        ctx.clear(0, 0, 0)

        chunks_to_remove = []
        unload_dist = camera.render_dist + 20

        for (cx, cy), chunk_data in chunk_vaos.items():
            dist_x = abs(cx - camera.chunk_row)
            dist_y = abs(cy - camera.chunk_col)

            if dist_x > unload_dist or dist_y > unload_dist:
                chunks_to_remove.append((cx, cy))

        for cx, cy in chunks_to_remove:
            chunk_data = chunk_vaos.pop((cx, cy))
            world.chunk_voxels.pop((cx, cy), None)
            world.requested_chunks.discard((cx, cy))

            if 'vao' in chunk_data: chunk_data['vao'].release()
            if 'vbo' in chunk_data: chunk_data['vbo'].release()
            if 'low_detail_vao' in chunk_data: chunk_data['low_detail_vao'].release()
            if 'low_detail_vbo' in chunk_data: chunk_data['low_detail_vbo'].release()
            if 'trans_vao' in chunk_data: chunk_data['trans_vao'].release()
            if 'trans_vbo' in chunk_data: chunk_data['trans_vbo'].release()
            if 'trans_lod_vao' in chunk_data: chunk_data['trans_lod_vao'].release()
            if 'trans_lod_vbo' in chunk_data: chunk_data['trans_lod_vbo'].release()

        view = camera.get_view_matrix()

        prog['u_view'].write(view.to_bytes())
        chunks_generated_this_frame = 0

        for cx in range(camera.chunk_row - (camera.render_dist + 10), camera.chunk_row + camera.render_dist + 10):
            for cy in range(camera.chunk_col - (camera.render_dist + 10), camera.chunk_col + camera.render_dist + 10):
                if (cx, cy) not in chunk_vaos:
                    world.request_chunk(cx, cy)

        chunks_uploaded_this_frame = 0
        while not world.result_queue.empty() and chunks_uploaded_this_frame < 3:
            (cx, cy), meshes, voxels = world.result_queue.get()
            mesh_data, low_res_mesh_data, transparencies_mesh_data, low_res_transparencies_mesh_data = meshes
            world.chunk_voxels[(cx, cy)] = voxels

            chunk_entry: dict[str, Any] = {
                'count': len(mesh_data),
                'low_detail_count': len(low_res_mesh_data),
                'trans_count': len(transparencies_mesh_data),
                'trans_lod_count': len(low_res_transparencies_mesh_data)
            }

            if len(mesh_data) > 0:
                vbo = ctx.buffer(mesh_data.tobytes())
                chunk_entry['vbo'] = vbo
                chunk_entry['vao'] = ctx.vertex_array(prog,
                                                      [(vbo, '3f 3f 1f', 'in_position', 'in_normal', 'in_material')])

            if len(low_res_mesh_data) > 0:
                low_vbo = ctx.buffer(low_res_mesh_data.tobytes())
                chunk_entry['low_detail_vbo'] = low_vbo
                chunk_entry['low_detail_vao'] = ctx.vertex_array(prog,
                                                                 [(low_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                   'in_material')])

            if len(transparencies_mesh_data) > 0:
                trans_vbo = ctx.buffer(transparencies_mesh_data.tobytes())
                chunk_entry['trans_vbo'] = trans_vbo
                chunk_entry['trans_vao'] = ctx.vertex_array(prog, [(trans_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                    'in_material')])

            if len(low_res_transparencies_mesh_data) > 0:
                trans_lod_vbo = ctx.buffer(low_res_transparencies_mesh_data.tobytes())
                chunk_entry['trans_lod_vbo'] = trans_lod_vbo
                chunk_entry['trans_lod_vao'] = ctx.vertex_array(prog,
                                                                [(trans_lod_vbo, '3f 3f 1f', 'in_position', 'in_normal',
                                                                  'in_material')])

            chunk_vaos[(cx, cy)] = chunk_entry

            chunk_count += 1
            chunks_uploaded_this_frame += 1

        for (cx, cy), chunk_data in chunk_vaos.items():
            dist_x = abs(cx - camera.chunk_row)
            dist_y = abs(cy - camera.chunk_col)

            if dist_x <= camera.render_dist and dist_y <= camera.render_dist:
                if chunk_data.get('count', 0) > 0 and 'vao' in chunk_data:
                    chunk_data['vao'].render(moderngl.TRIANGLES, vertices=chunk_data['count'])

            elif dist_x <= camera.render_dist + 20 and dist_y <= camera.render_dist + 20:
                if chunk_data.get('low_detail_count', 0) > 0 and 'low_detail_vao' in chunk_data:
                    chunk_data['low_detail_vao'].render(moderngl.TRIANGLES, vertices=chunk_data['low_detail_count'])

        ctx.depth_mask = False

        for (cx, cy), chunk_data in chunk_vaos.items():
            dist_x = abs(cx - camera.chunk_row)
            dist_y = abs(cy - camera.chunk_col)

            if dist_x <= camera.render_dist and dist_y <= camera.render_dist:
                if chunk_data.get('trans_count', 0) > 0 and 'trans_vao' in chunk_data:
                    chunk_data['trans_vao'].render(moderngl.TRIANGLES, vertices=chunk_data['trans_count'])

            elif dist_x <= camera.render_dist + 20 and dist_y <= camera.render_dist + 20:
                if chunk_data.get('trans_lod_count', 0) > 0 and 'trans_lod_vao' in chunk_data:
                    chunk_data['trans_lod_vao'].render(moderngl.TRIANGLES, vertices=chunk_data['trans_lod_count'])
                    
        ctx.depth_mask = True

    keys = pygame.key.get_pressed()
    yaw_rad = math.radians(camera.yaw)
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.sin(yaw_rad)
    right_y = -math.cos(yaw_rad)
    speed = 25.0 * dt
    moving = False
    if keys[pygame.K_w]:
        camera.X += forward_x * speed
        camera.Y += forward_y * speed
        moving = True
    if keys[pygame.K_s]:
        camera.X -= forward_x * speed
        camera.Y -= forward_y * speed
        moving = True
    if keys[pygame.K_d]:
        camera.X += right_x * speed
        camera.Y += right_y * speed
        moving = True
    if keys[pygame.K_a]:
        camera.X -= right_x * speed
        camera.Y -= right_y * speed
        moving = True
    camera.physics_calc(speed, moving, world, direction=(forward_x, forward_y, right_x, right_y), key=keys)
    camera.get_chunk()

    dx, dy = pygame.mouse.get_rel()
    camera.yaw -= dx * sensitivity
    camera.pitch -= dy * sensitivity

    camera.pitch = max(-89, min(89, camera.pitch))

    mouse = pygame.mouse.get_pressed()
    block = world.get_picked_block(1000, 800, camera.get_view_matrix(), camera.projection_matrix(1000 / 800),
                                   (camera.X, camera.Y, camera.Z), dist=20)
    if block:
        draw.text(f'{math.floor(block['hit_pos'][0]), math.floor(block['hit_pos'][1]), math.floor(block['hit_pos'][2])}'
                  f'{block['material']}, '
                  f'{block['chunk']}', font_size=15, position=(10, 400))

        if mouse[2]:
            modified_chunks = world.dig(3, block['hit_pos'][0], block['hit_pos'][1], block['hit_pos'][2])

            for (cx, cy), (meshes, voxels) in modified_chunks.items():
                world.rebuild_gpu_chunk(cx, cy, meshes, chunk_vaos, ctx, prog)

    fps.update()
    draw.text(str(fps.frame_push), position=(10, 10))

    draw.rect((490, 390), (10, 10), (0, 0, 0))
    pygame.display.flip()

pygame.quit()
sys.exit()