import sys
from math import copysign, inf

import pygame

from vectors import Vector2D

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return Vector2D(x, y)

def project_polygon(vertices, axis):
    """다각형을 축에 투영하여 최소/최대 범위 반환."""
    projections = [v.dot(axis) for v in vertices]
    return min(projections), max(projections)

def overlap(range1, range2):
    """두 범위가 겹치는지 확인."""
    return range1[0] <= range2[1] and range2[0] <= range1[1]

def sat_collision(body1, body2):
    """SAT를 이용한 충돌 감지. 원의 특수 처리를 포함."""
    axes = []
    minimum_depth = float('inf')  # 초기 충돌 깊이를 무한대로 설정
    collision_axis = None

    # 다각형의 축 (법선 벡터)
    if body1.shape in ["rectangle", "triangle"]:
        axes += [edge.orthogonal().normalize() for edge in body1.edges]
    if body2.shape in ["rectangle", "triangle"]:
        axes += [edge.orthogonal().normalize() for edge in body2.edges]

    # 원과 다각형 간 충돌 축 추가
    if body1.shape == "circle" or body2.shape == "circle":
        if body1.shape == "circle" and body2.shape == "circle":
            # 원과 원 간 충돌 축
            axis = (body2.position - body1.position).normalize()
            axes.append(axis)
        else:
            # 원과 다각형 간 충돌 축
            circle_body = body1 if body1.shape == "circle" else body2
            polygon_body = body2 if body1.shape == "circle" else body1
            closest_vertex = min(polygon_body.vertices, key=lambda v: (v - circle_body.position).length())
            axis = (closest_vertex - circle_body.position).normalize()
            axes.append(axis)

    # 모든 축에 대해 투영 확인
    for axis in axes:
        # 다각형 투영 범위 계산
        range1 = project_polygon(body1.vertices, axis) if body1.shape != "circle" else (
            body1.position.dot(axis) - body1.width / 2,
            body1.position.dot(axis) + body1.width / 2,
        )
        range2 = project_polygon(body2.vertices, axis) if body2.shape != "circle" else (
            body2.position.dot(axis) - body2.width / 2,
            body2.position.dot(axis) + body2.width / 2,
        )

        # 투영 범위 겹침 확인
        if not overlap(range1, range2):
            return False, 0, None  # 충돌하지 않으면 깊이와 축 없음

        # 겹침 거리 계산
        overlap_depth = min(range1[1], range2[1]) - max(range1[0], range2[0])
        if overlap_depth < minimum_depth:
            minimum_depth = overlap_depth
            collision_axis = axis

    return True, minimum_depth, collision_axis

class RigidBody:
    def __init__(self, width, height, x, y, angle=0.0, mass=None, restitution=0.5, shape="rectangle"):
        self.position = Vector2D(x, y)
        self.width = width
        self.height = height
        self.angle = angle
        self.shape = shape
        self.velocity = Vector2D(0.0, 0.0)
        self.angular_velocity = 0.0

        self.torque = 0.0
        self.forces = Vector2D(0.0, 0.0)
        if mass is None:
            mass = width * height
        self.mass = mass
        self.restitution = restitution
        self.inertia = mass * (width ** 2 + height ** 2) / 12

        self.sprite = pygame.Surface((width, height))
        self.sprite.set_colorkey((0, 0, 0))
        self.sprite.fill((0, 0, 0))
        pygame.draw.rect(self.sprite, (255, 255, 255), (0, 0, width - 2, height - 2), 2)

    def draw(self, surface):
        if self.shape == "circle":
            pygame.draw.circle(surface, (255, 255, 255), (int(self.position.x), int(self.position.y)), self.width // 2, 2)
        elif self.shape == "triangle":
            # 삼각형 꼭짓점 계산
            half_width = self.width / 2
            half_height = self.height / 2
            points = [
                (0, -half_height),  # 위쪽 꼭짓점
                (-half_width, half_height),  # 왼쪽 아래 꼭짓점
                (half_width, half_height),  # 오른쪽 아래 꼭짓점
            ]
            # 꼭짓점을 회전 및 이동
            rotated_points = [Vector2D(p).rotate(-self.angle) + self.position for p in points]
            pygame.draw.polygon(surface, (255, 255, 255), [(p.x, p.y) for p in rotated_points], 2)
        else:
            rotated = pygame.transform.rotate(self.sprite, self.angle)
            rect = rotated.get_rect()
            surface.blit(rotated, self.position - (rect.width / 2, rect.height / 2))

    def add_world_force(self, force, offset):

        if abs(offset[0]) <= self.width / 2 and abs(offset[1]) <= self.height / 2:
            self.forces += force
            self.torque += offset.cross(force.rotate(self.angle))

    def add_torque(self, torque):
        self.torque += torque

    def apply_friction(self, friction_coefficient):
        if self.velocity.length() > 0:
            friction_force = -self.velocity.normalize() * (friction_coefficient * self.mass * 9.81)
            self.forces += friction_force

    def reset(self):
        self.forces = Vector2D(0.0, 0.0)
        self.torque = 0.0

    def update(self, dt):
        self.apply_friction(friction_coefficient=75) # 마찰계수
        rotational_drag_coefficient = 0.5 # 회전저항
        self.angular_velocity *= (1 - rotational_drag_coefficient * dt)

        acceleration = self.forces / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        angular_acceleration = self.torque / self.inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        # 속도 임계값 확인 (속도가 작으면 멈춤)
        velocity_threshold = 12  # 속도 임계값
        if self.velocity.length() < velocity_threshold:
            self.velocity = Vector2D(0, 0)  # 속도 0으로 설정
        # 각속도 임계값 확인 (회전 멈춤)
        if abs(self.angular_velocity) < velocity_threshold:
            self.angular_velocity = 0

        self.reset()

    @property
    def vertices(self):
        if self.shape == "rectangle":
            return [
                self.position + Vector2D(v).rotate(-self.angle) for v in (
                    (-self.width / 2, -self.height / 2),
                    (self.width / 2, -self.height / 2),
                    (self.width / 2, self.height / 2),
                    (-self.width / 2, self.height / 2),
                )
            ]
        elif self.shape == "triangle":
            half_width = self.width / 2
            half_height = self.height / 2
            return [
                self.position + Vector2D(v).rotate(-self.angle) for v in (
                    (0, -half_height),  # 위쪽
                    (-half_width, half_height),  # 왼쪽 아래
                    (half_width, half_height),  # 오른쪽 아래
                )
            ]
        elif self.shape == "circle":
            # 원은 꼭짓점 대신 중심만 반환
            return [self.position]
        return []

    @property
    def edges(self):
        if self.shape in ["rectangle", "triangle"]:
            vertices = self.vertices
            return [vertices[i] - vertices[i - 1] for i in range(len(vertices))]
        return []

    def collide(self, other):
        # Exit early for optimization
        if (self.position - other.position).length() > max(self.width, self.height) + max(other.width, other.height):
            return False, None, None

        def project(vertices, axis):
            dots = [vertex.dot(axis) for vertex in vertices]
            return Vector2D(min(dots), max(dots))

        collision_depth = sys.maxsize
        collision_normal = None

        for edge in self.edges + other.edges:
            axis = Vector2D(edge).orthogonal().normalize()
            projection_1 = project(self.vertices, axis)
            projection_2 = project(other.vertices, axis)
            min_intersection = max(min(projection_1), min(projection_2))
            max_intersection = min(max(projection_1), max(projection_2))
            overlapping = min_intersection <= max_intersection
            if not overlapping:
                return False, None, None
            else:
                overlap = max_intersection - min_intersection
                if overlap < collision_depth:
                    collision_depth = overlap
                    collision_normal = axis
        return True, collision_depth, collision_normal

    def get_collision_edge(self, normal):
        max_projection = -sys.maxsize
        support_point = None
        vertices = self.vertices
        length = len(vertices)

        for i, vertex in enumerate(vertices):
            projection = vertex.dot(normal)
            if projection > max_projection:
                max_projection = projection
                support_point = vertex
                if i == 0:
                    right_vertex = vertices[-1]
                else:
                    right_vertex = vertices[i - 1]
                if i == length - 1:
                    left_vertex = vertices[0]
                else:
                    left_vertex = vertices[i + 1]

        if right_vertex.dot(normal) > left_vertex.dot(normal):
            return (right_vertex, support_point)
        else:
            return (support_point, left_vertex)


class PhysicsWorld:
    def __init__(self):
        self.bodies = []

    def add(self, *bodies):
        self.bodies += bodies
        for body in bodies:
            print("Body added", id(body))

    def remove(self, body):
        self.bodies.remove(body)
        print("Body removed", id(body))

    def update(self, dt):
        tested = []
        for body in self.bodies:
            for other_body in self.bodies:
                if other_body not in tested and other_body is not body:
                    collision, depth, normal = sat_collision(body, other_body) # SAT 충돌 감지 호출
                    if collision:
                        print(f"Collision detected between {body} and {other_body}")
                        # 충돌 처리
                        rel_vel = (body.velocity - other_body.velocity)
                        j = -(1 + body.restitution) * rel_vel.dot(normal) / normal.dot(
                            normal * (1 / body.mass + 1 / other_body.mass))

                        direction = body.position - other_body.position
                        magnitude = normal.dot(direction)

                        if body.mass != inf:
                            body.position += normal * depth * copysign(1, magnitude)
                        if other_body.mass != inf:
                            other_body.position -= normal * depth * copysign(1, magnitude)

                        body.velocity = body.velocity + j / body.mass * normal
                        other_body.velocity = other_body.velocity - j / other_body.mass * normal

                        # 충돌 접점 계산
                        body_collision_edge = body.get_collision_edge(-direction)
                        other_body_collision_edge = other_body.get_collision_edge(direction)
                        # 접촉점 계산
                        if abs(normal.dot(direction)) < 1e-3:  # 충돌면이 평행한 경우
                            contact_point = (body.position + other_body.position) / 2  # 중심으로 설정
                        else:
                            contact_point = line_intersection(body_collision_edge, other_body_collision_edge)
                        if contact_point:
                            # 접점에서의 충격량 계산
                            impulse = j * normal
                        
                            # 접점에서의 토크 계산 및 각속도 업데이트
                            r_body = (contact_point - body.position)
                            r_other_body = (contact_point - other_body.position)

                            if body.mass != inf:
                                torque_body = -r_body.cross(impulse)
                                body.angular_velocity += torque_body / body.inertia
                            if other_body.mass != inf:
                                torque_other = -r_other_body.cross(impulse)
                                other_body.angular_velocity -= torque_other / other_body.inertia

            tested.append(body)

            # 물체 업데이트 (위치, 속도, 각속도 변경 적용)
            body.update(dt)