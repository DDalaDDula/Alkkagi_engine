## Alkkagi Engine

해당 프로젝트는 **Pygame**을 사용해 구현된 **알까기 시뮬레이터**입니다. 강체의 충돌, 회전, 마찰 등 기본적인 물리 시뮬레이션을 지원하며, 사용자 인터페이스를 통해 직사각형, 원, 삼각형 등의 객체를 추가하고 상호작용할 수 있습니다.

### **Getting Started**
- Python 3.8 or higher
- Pygame

## **Features**

### 1. **Collision Detection**
- Implements the **Separating Axis Theorem (SAT)** for accurate collision detection.
- Handles collisions between:
  - Rectangle vs Rectangle
  - Circle vs Circle
  - Rectangle vs Circle
  - Triangle vs other shapes

### 2. **Physics Simulation**
- Supports rigid body dynamics with:
  - Linear velocity and acceleration
  - Angular velocity and torque
  - Friction and rotational drag
- Handles realistic collision responses, including:
  - Impulse-based collision resolution
  - Position correction to avoid overlaps

### 3. **Interactive Drag-and-Drop**
- Add objects dynamically via drag-and-drop with:
  - Adjustable velocity based on drag distance.
  - Real-time preview of the object’s orientation.

### 4. **Object Management**
- Automatically removes objects that move completely outside the screen boundaries

---

### **사용 방법 및 기본 조작**
- **객체 추가**:
  - 화면 오른쪽 상단의 버튼(직사각형, 원, 삼각형)을 클릭하여 도형 선택.
  - 화면에서 드래그하여 객체를 추가.
- **물리 시뮬레이션**:
  - 추가된 객체는 중력, 충돌, 회전을 포함한 물리 법칙에 따라 동작.
- **FPS 및 객체 상태 표시**:
  - 화면 왼쪽 상단에서 FPS와 활성 객체 수를 확인.

### **키보드 컨트롤**
- **ESC**: 프로그램 종료
- **마우스**:
  - 드래그 앤 드롭으로 객체 추가.
  - 도형 선택은 화면 버튼을 사용.

---

### **코드 구조(모듈)**
1. **`vectors.py`**:
   - 2D 벡터 연산 (덧셈, 뺄셈, 크로스 곱, 회전 등) 구현.

2. **`physics.py`**:
   - `RigidBody`와 `PhysicsWorld` 클래스 포함:
     - `RigidBody`: 개별 물체의 물리적 특성과 동작을 정의.
     - `PhysicsWorld`: 물체 간 상호작용 및 충돌을 관리.

3. **`game.py`**:
   - 사용자 인터페이스와 렌더링, 게임 루프 처리.

---

### **주요 클래스**

### 1. **`Vector2D`**
- 2D 벡터 연산을 위한 클래스.
- 주요 메서드:
  - `length()`: 벡터 크기 계산.
  - `rotate(theta)`: 벡터를 특정 각도 `theta`만큼 회전.
  - `dot(other)`: 두 벡터의 내적 계산.
  - `cross(other)`: 두 벡터의 외적 계산.

### 2. **`RigidBody`**
- 개별 물체의 물리적 특성을 정의하는 클래스.
- 주요 기능:
  - 질량, 속도, 각속도, 모양(직사각형, 원, 삼각형 등)을 설정.
  - 충돌 처리 및 물리 업데이트 지원.
- 주요 메서드:
  - `update(dt)`: 물체의 위치와 속도를 갱신.
  - `get_collision_edge(direction)`: 지정된 방향에 따른 충돌면 반환.

### 3. **`PhysicsWorld`**
- 물리 엔진의 핵심으로 모든 물체의 물리적 상호작용을 관리.
- 주요 메서드:
  - `add(body)`: 물체를 월드에 추가.
  - `update(dt)`: 모든 물체를 업데이트하고 충돌 처리.

---
