import traceback
import pygame
from pygame.locals import *
from vectors import Vector2D
from physics import PhysicsWorld, RigidBody
from math import atan2, degrees

pygame.display.init()
pygame.font.init()
pygame.display.set_caption("Alkkagi engine")
default_font = pygame.font.Font(None, 24)
screen_size = (900, 900)
game_surface = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

world = PhysicsWorld()
mouse_pos = Vector2D(screen_size) / 2

# 드래그 상태변수
is_dragging = False
drag_start_pos = None
mouse_pos = None  # 마우스 현재 위치
dragging_body = None  # 드래그 중인 물체

background_image = pygame.image.load('./img/Board.png')
background_image = pygame.transform.scale(background_image, screen_size)

# 물체 형태 상태
shape_type = "rectangle"  # 초기 형태 - 사각형
color_type = "black"

# 물체 형태 선택버튼 정보
buttons = [
    {"label": "Rectangle", "x": screen_size[0] - 350, "y": 10, "width": 100, "height": 30},
    {"label": "Circle", "x": screen_size[0] - 230, "y": 10, "width": 100, "height": 30},
    {"label": "Triangle", "x": screen_size[0] - 110, "y": 10, "width": 100, "height": 30},
    {"label": "Black", "x": screen_size[0] - 110, "y": screen_size[1] - 50, "width": 50, "height": 30},
    {"label": "White", "x": screen_size[0] - 60, "y": screen_size[1] - 50, "width": 50, "height": 30},
]


def draw_button():
    for button in buttons:
        rect = pygame.Rect(button["x"], button["y"], button["width"], button["height"])
        if button["label"] == "Black":
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
        elif button["label"] == "White":
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)
        else:
            bg_color = (200, 200, 200)
            text_color = (0, 0, 0)
        pygame.draw.rect(game_surface, bg_color, rect)
        pygame.draw.rect(game_surface, (100, 100, 100), rect, 2)
        text_surface = default_font.render(button["label"], True, text_color)
        text_rect = text_surface.get_rect(center=(button["x"] + button["width"] // 2, button["y"] + button["height"] // 2))
        game_surface.blit(text_surface, text_rect)


def is_mouse_on_button(pos):
    for button in buttons:
        if button["x"] <= pos[0] <= button["x"] + button["width"] and button["y"] <= pos[1] <= button["y"] + button["height"]:
            return button  # 클릭된 버튼
    return None


# 버튼 클릭 처리
def check_button_click(button):
    global shape_type, color_type
    if button["label"] in ["Rectangle", "Circle", "Triangle"]:
        shape_type = button["label"].lower()
        print(f"Shape type changed to {shape_type}.")
    elif button["label"] in ["Black", "White"]:
        color_type = button["label"].lower()
        print(f"Color type changed to {color_type}.")


def get_input():
    global is_dragging, dragging_body, drag_start_pos, mouse_pos, shape_type, color_type
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # 버튼 클릭 확인
            clicked_button = is_mouse_on_button(event.pos)
            if clicked_button:
                check_button_click(clicked_button)
                return True  # 버튼 클릭 이벤트는 여기서 종료

            drag_start_pos = Vector2D(mouse_pos)
            # 클릭한 위치에 있는 물체 감지
            dragging_body = world.get_body_at_position(drag_start_pos)
            if dragging_body:
                is_dragging = True  # 기존 물체 드래그 시작
                # 마우스 위치를 물체 중심으로 보정
                drag_start_pos = dragging_body.position
            else:
                is_dragging = True  # 새로운 물체 생성 준비
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if is_dragging and dragging_body:
                # 기존 물체를 발사
                drag_end_pos = Vector2D(mouse_pos)
                direction_vector = drag_start_pos - drag_end_pos
                dragging_body.velocity = direction_vector * 2.5  # 발사 속도
                dragging_body = None  # 드래그 상태 초기화
                is_dragging = False
            elif is_dragging:
                # 새로운 물체 생성
                is_dragging = False
                drag_end_pos = Vector2D(mouse_pos)
                direction_vector = drag_start_pos - drag_end_pos
                launch_velocity = direction_vector * 2.5
                angle = degrees(atan2(-direction_vector.y, direction_vector.x)) if direction_vector.length() > 0 else 0

                # 위치 점유 확인
                position = drag_start_pos
                width, height = 40, 40
                if world.is_position_occupied(position, shape_type, width, height):
                    print("Cannot create object here: position is already occupied.")
                    return True  # 위치가 차있으면 물체 생성하지 않음

                # 색상 설정
                color = (0, 0, 0) if color_type == "black" else (255, 255, 255)

                # 물체 생성
                if shape_type == "rectangle":
                    body = RigidBody(width, height, position.x, position.y, angle=angle, color=color)
                elif shape_type == "circle":
                    body = RigidBody(width, height, position.x, position.y, angle=angle, shape="circle", color=color)
                elif shape_type == "triangle":
                    body = RigidBody(width, height, position.x, position.y, angle=angle - 90, shape="triangle", color=color)

                body.velocity = launch_velocity
                world.add(body)
        elif event.type == pygame.MOUSEMOTION and is_dragging and dragging_body:
            # 기존 물체를 드래그 중
            dragging_body.position = drag_start_pos  # 물체를 드래그 시작 위치에 고정
    return True

def draw_preview():
    if is_dragging and drag_start_pos and not dragging_body:
        # 방향 벡터 계산
        direction_vector = Vector2D(mouse_pos) - drag_start_pos
        angle = degrees(atan2(-direction_vector.y, direction_vector.x))

        # 도형 미리보기 그리기
        color = (0, 0, 0) if color_type == "black" else (255, 255, 255)
        if shape_type == "rectangle":
            rect = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.rect(rect, color, (0, 0, 40, 40), 100)
            rotated_rect = pygame.transform.rotate(rect, angle)
            rect_pos = (drag_start_pos.x - rotated_rect.get_width() // 2, drag_start_pos.y - rotated_rect.get_height() // 2)
            game_surface.blit(rotated_rect, rect_pos)
        elif shape_type == "circle":
            pygame.draw.circle(game_surface, color, (int(drag_start_pos.x), int(drag_start_pos.y)), 20)
        elif shape_type == "triangle":
            # 삼각형(화살표) 꼭짓점 정의
            points = [(-20, 0), (20, 20), (20, -20)]
            rotated_points = [Vector2D(p).rotate(-angle) + drag_start_pos for p in points]
            pygame.draw.polygon(game_surface, color, [(p.x, p.y) for p in rotated_points])

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
    game_surface.blit(background_image, (0, 0))
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
