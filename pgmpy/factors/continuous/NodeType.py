from enum import Enum

class NodeType(Enum):
    GAUSSIAN = 0
    SPBN = 1
    SPBN_STRICT = 2

    @classmethod
    def str(cls, n):
        if n == cls.GAUSSIAN:
            return "GAUSSIAN"
        elif n == cls.SPBN:
            return "SPBN"
        elif n == cls.SPBN_STRICT:
            return "SPBN STRICT"
        else:
            raise ValueError("Value not valid for NodeType")