class NodeType:
    GAUSSIAN = 0
    CKDE = 1

    @classmethod
    def str(cls, n):
        if n == cls.GAUSSIAN:
            return "GAUSSIAN"
        elif n == cls.CKDE:
            return "CKDE"
        else:
            raise ValueError("Value not valid for NodeType")