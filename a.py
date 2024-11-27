import sys
import random
import traceback
import pygame
from pygame.locals import *
from math import cos, sin, sqrt, radians, copysign, inf

class Vector2D:
    def __init__(self, *args):
        if args.__len__() == 2:
            self.x, self.y = args[0], args[1]
        elif args.__len__() == 1:
            self.x, self.y = args[0]
        else:
            self.x, self.y = 0.0, 0.0

    def __add__(self, other):
        if len(other) == len(self):
            return self.__class__(*(a + b for a, b in zip(self, other)))
        else:
            raise TypeError

    def __sub__(self, other):
        if len(other) == len(self):
            return self.__class__(*(a - b for a, b in zip(self, other)))
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.x * other, self.y * other)
        else:
            raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.x / other, self.y / other)
        else:
            raise TypeError

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __len__(self):
        return 2

    def __getitem__(self, key):
        return (self.x, self.y)[key]

    def __repr__(self):
        return "{} ({}, {})".format(self.__class__.__name__, self.x, self.y)

    def length(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        length = self.length()
        return self.__class__(self.x / length, self.y / length)

    def rotate(self, theta):
        theta = radians(theta)
        dc, ds = cos(theta), sin(theta)
        x, y = dc * self.x - ds * self.y, ds * self.x + dc * self.y
        return self.__class__(x, y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def orthogonal(self):
        return self.__class__(self.x, -self.y)

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


class RigidBody:
    def __init__(self, width, height, x, y, angle=0.0, mass=None, restitution=0.5):
        self.position = Vector2D(x, y)
        self.width = width
        self.height = height
        self.angle = angle

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
        rotated = pygame.transform.rotate(self.sprite, self.angle)
        rect = rotated.get_rect()
        surface.blit(rotated, self.position - (rect.width / 2, rect.height / 2))

    def add_world_force(self, force, offset):

        if abs(offset[0]) <= self.width / 2 and abs(offset[1]) <= self.height / 2:
            self.forces += force
            self.torque += offset.cross(force.rotate(self.angle))

    def add_torque(self, torque):
        self.torque += torque

    def reset(self):
        self.forces = Vector2D(0.0, 0.0)
        self.torque = 0.0

    def update(self, dt):
        acceleration = self.forces / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        angular_acceleration = self.torque / self.inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        self.reset()

    @property
    def vertices(self):
        return [
            self.position + Vector2D(v).rotate(-self.angle) for v in (
                (-self.width / 2, -self.height / 2),
                (self.width / 2, -self.height / 2),
                (self.width / 2, self.height / 2),
                (-self.width / 2, self.height / 2)
            )
        ]

    @property
    def edges(self):
        return [
            Vector2D(v).rotate(self.angle) for v in (
                (self.width, 0),
                (0, self.height),
                (-self.width, 0),
                (0, -self.height),
            )
        ]

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
                    collision, depth, normal = body.collide(other_body)

                    if collision:
                        normal = normal.normalize()

                        rel_vel = (body.velocity - other_body.velocity)
                        j = -(1 + body.restitution) * rel_vel.dot(normal) / normal.dot(
                            normal * (1 / body.mass + 1 / other_body.mass))

                        direction = body.position - other_body.position
                        magnitude = normal.dot(direction)

                        if body.mass != inf:
                            body.position += normal * depth * copysign(1, magnitude)
                        if other_body.mass != inf:
                            other_body.position -= normal * depth * copysign(1, magnitude)
                        
                        friction_coefficient = 0.5  # 마찰 계수
                        tangent = normal.orthogonal() # 기존 충격량 계산에 마찰력 추가
                        friction_impulse = -(rel_vel.dot(tangent)) * friction_coefficient / (1 / body.mass + 1 / other_body.mass)

                        body.velocity += (normal * j + tangent * friction_impulse) / body.mass
                        other_body.velocity -= (normal * j + tangent * friction_impulse) / other_body.mass

                        body_collision_edge = body.get_collision_edge(-direction)
                        other_body_collision_edge = other_body.get_collision_edge(direction)
                        contact_point = line_intersection(body_collision_edge, other_body_collision_edge)

                        if contact_point:
                            radius = (body.position - contact_point)
                            body.angular_velocity = body.angular_velocity + (radius.dot(j * normal / body.inertia))

                            radius = (other_body.position - contact_point)
                            other_body.angular_velocity = other_body.angular_velocity - (
                                radius.dot(j * normal / other_body.inertia))

            tested.append(body)
            body.update(dt)

pygame.display.init()
pygame.font.init()
pygame.display.set_caption("Simple physics example")
default_font = pygame.font.Font(None, 24)
screen_size = (1280, 768)
game_surface = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

world = PhysicsWorld()
world.add(
    RigidBody(100, 100, 100, 100, mass=inf),
    RigidBody(100, 100, screen_size[0] - 100, 100, mass=inf),
    RigidBody(100, 100, screen_size[0] - 100, screen_size[1] - 100, mass=inf),
    RigidBody(100, 100, 100, screen_size[1] - 100, mass=inf),
)
screen_center = Vector2D(screen_size) / 2
mouse_pos = screen_center


def get_input():
    mouse_buttons = pygame.mouse.get_pressed()
    global mouse_pos
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
        elif event.type == pygame.MOUSEBUTTONUP and mouse_buttons[0]:
            body = RigidBody(
                50, 50,
                screen_center.x, screen_center.y,
                angle=random.randint(0, 90)
            )
            world.add(body)
            body.velocity = Vector2D(mouse_pos) - screen_center
    return True


def draw():
    game_surface.fill((40, 40, 40))

    for body in world.bodies:
        body.draw(game_surface)
    pygame.draw.line(game_surface, (0, 255, 0), screen_center, mouse_pos, 2)

    game_surface.blit(default_font.render('Objects: {}'.format(len(world.bodies)), True, (255, 255, 255)), (0, 0))
    game_surface.blit(default_font.render('FPS: {0:.0f}'.format(clock.get_fps()), True, (255, 255, 255)), (0, 24))
    pygame.display.update()


def main():
    dt = 1 / 60
    while True:
        if not get_input():
            break
        world.update(dt)
        for body in world.bodies:
            if body.position.x < 0 or body.position.x > screen_size[0] or \
                    body.position.y < 0 or body.position.y > screen_size[1]:
                world.remove(body)
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
