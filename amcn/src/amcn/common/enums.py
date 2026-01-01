from enum import IntEnum

class Scene(IntEnum):
    STATIC_OPEN = 0
    STATIC_CROWD = 1
    STATIC_NARROW = 2

    DYNAMIC_OPEN = 3
    DYNAMIC_CROWD = 4

    RECOVERY = 5
