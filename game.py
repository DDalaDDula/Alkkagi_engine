import random
import traceback
import pygame
from pygame.locals import *
from vectors import Vector2D
from physics import PhysicsWorld, RigidBody
from math import atan2, degrees

pygame.display.init()
pygame.font.init()
pygame.display.set_caption("physics engine")
default_font = pygame.font.Font(None, 24)
screen_size = (1280, 768)
game_surface = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

world = PhysicsWorld()
mouse_pos = Vector2D(screen_size) / 2

# 드래그 상태변수
is_dragging = False
drag_start_pos = None
mouse_pos = None  # 마우스 현재 위치

# 물체 형태 상태
shape_type = "rectangle"  # 초기 형태 - 사각형

# 물체 형태 선택버튼 정보
buttons = [
    {"label": "Rectangle", "x": screen_size[0] - 350, "y": 10, "width": 100, "height": 30},
    {"label": "Circle", "x": screen_size[0] - 230, "y": 10, "width": 100, "height": 30},
    {"label": "Triangle", "x": screen_size[0] - 110, "y": 10, "width": 100, "height": 30},
]


def draw_button():
    for button in buttons:
        rect = pygame.Rect(button["x"], button["y"], button["width"], button["height"])
        pygame.draw.rect(game_surface, (200, 200, 200), rect)
        pygame.draw.rect(game_surface, (100, 100, 100), rect, 2)
        text_surface = default_font.render(button["label"], True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(button["x"] + button["width"] // 2, button["y"] + button["height"] // 2))
        game_surface.blit(text_surface, text_rect)


def is_mouse_on_button(pos):
    for button in buttons:
        if button["x"] <= pos[0] <= button["x"] + button["width"] and button["y"] <= pos[1] <= button["y"] + button["height"]:
            return True
    return False


def check_button_click(pos):
    global shape_type
    for button in buttons:
        if button["x"] <= pos[0] <= button["x"] + button["width"] and button["y"] <= pos[1] <= button["y"] + button["height"]:
            shape_type = button["label"].lower()  # "rectangle", "circle", "triangle"


def get_input():
    global is_dragging, drag_start_pos, mouse_pos, shape_type
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # 버튼 클릭 시 드래그 시작 차단
            if is_mouse_on_button(event.pos):
                check_button_click(event.pos)
                return True
            # 드래그 시작
            is_dragging = True
            drag_start_pos = Vector2D(mouse_pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and is_dragging:
            # 드래그 종료 및 발사
            is_dragging = False
            drag_end_pos = Vector2D(mouse_pos)
            direction_vector = drag_start_pos - drag_end_pos
            launch_velocity = direction_vector * 2.5  # 속도 감도 조정
            angle = degrees(atan2(-direction_vector.y, direction_vector.x)) if direction_vector.length() > 0 else 0

            if shape_type == "rectangle":
                body = RigidBody(40, 40, drag_start_pos.x, drag_start_pos.y, angle=angle)
            elif shape_type == "circle":
                body = RigidBody(40, 40, drag_start_pos.x, drag_start_pos.y, angle=angle, shape="circle")
            elif shape_type == "triangle":
                body = RigidBody(40, 40, drag_start_pos.x, drag_start_pos.y, angle=angle - 90, shape="triangle")

            body.velocity = launch_velocity
            world.add(body)
    return True

def draw_preview():
    if is_dragging and drag_start_pos:
        # 방향 벡터 계산
        direction_vector = Vector2D(mouse_pos) - drag_start_pos
        angle = degrees(atan2(-direction_vector.y, direction_vector.x))

        # 도형 미리보기 그리기
        if shape_type == "rectangle":
            rect = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.rect(rect, (200, 200, 200), (0, 0, 40, 40), 2)
            rotated_rect = pygame.transform.rotate(rect, angle)
            rect_pos = (drag_start_pos.x - rotated_rect.get_width() // 2, drag_start_pos.y - rotated_rect.get_height() // 2)
            game_surface.blit(rotated_rect, rect_pos)
        elif shape_type == "circle":
            pygame.draw.circle(game_surface, (200, 200, 200), (int(drag_start_pos.x), int(drag_start_pos.y)), 20, 2)
        elif shape_type == "triangle":
            # 삼각형(화살표) 꼭짓점 정의
            points = [(-20, 0), (20, 20), (20, -20)]
            rotated_points = [Vector2D(p).rotate(-angle) + drag_start_pos for p in points]
            pygame.draw.polygon(game_surface, (200, 200, 200), [(p.x, p.y) for p in rotated_points], 2)

def remove_out_of_bounds_bodies(world, screen_width, screen_height):
    bodies_to_remove = []

    for body in world.bodies:
        left = body.position.x - body.width / 2
        right = body.position.x + body.width / 2
        top = body.position.y - body.height / 2
        bottom = body.position.y + body.height / 2

        if right < 0 or left > screen_width or bottom < 0 or top > screen_height:
            bodies_to_remove.append(body)

    for body in bodies_to_remove:
        world.remove(body)
        print(f"Removed body: {body}")


def draw():
    game_surface.fill((40, 40, 40))
    draw_button() # 버튼
    draw_preview() # 드래그 미리보기
    # 드래그 선
    if is_dragging and drag_start_pos:  
        pygame.draw.line(game_surface, (255, 0, 0), drag_start_pos, mouse_pos, 2)
    # 물체 그리기
    for body in world.bodies:
        body.draw(game_surface)
    # 화면 정보 표시
    game_surface.blit(default_font.render(f'Objects: {len(world.bodies)}', True, (255, 255, 255)), (10, 10))
    game_surface.blit(default_font.render(f'FPS: {clock.get_fps():.0f}', True, (255, 255, 255)), (10, 40))

    pygame.display.flip()


def main():
    dt = 1 / 60  # 60 FPS 기준
    while True:
        if not get_input():
            break
        world.update(dt)
        remove_out_of_bounds_bodies(world, screen_size[0], screen_size[1])
        draw()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        pygame.quit()
        input()
