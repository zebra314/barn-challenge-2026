from enum import Enum, auto

class MotionType(Enum):
    forward = auto()
    turn = auto()

    chicane = auto()
    threading = auto()

    brake = auto()
    spin = auto()
    backward = auto()
