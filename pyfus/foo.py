import numpy as np

class foo:
    _width: tuple[float] = (0.0,)
    _height: tuple[float] = (0.0,)
    units: str = 'm'
    
    def __init__(self, width=0, height=0, units='m'):
        self._parse_to_tuple('width', width, float)
        self._parse_to_tuple('height', height, float)
        propsizes = set(self._propsizes())
        propsizes.discard(1)
        if len(propsizes) > 1:
            raise ValueError("All tuples must be singleton or the same length")

    @property
    def width(self):
        if len(self._width) == 1:
            return self._width[0]
        else:
            return self._width

    @width.setter
    def width(self, value):
        self._parse_to_tuple('width', value, float)
    
    @property
    def height(self):
        if len(self._height) == 1:
            return self._height[0]
        else:
            return self._height

    @height.setter
    def height(self, value):
        self._parse_to_tuple('height', value, float)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                if isinstance(v, tuple) and len(v) == 1:
                    d[k[1:]] = v[0]
                else:
                    d[k[1:]] = v
            else:
                d[k] = v
        return d

    def _propsizes(self):
        propsizes = [len(v) for k, v in self.__dict__.items() if k.startswith('_')]
        if len(propsizes) == 0:
            propsizes = (1,)
        return propsizes

    def __len__(self):
        return max(self._propsizes())

    def __getitem__(self, index):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith('_') and isinstance(v, tuple):
                    d[k[1:]] = v[index]
            else:
                    d[k] = v
        return foo(**d)

    def _parse_to_tuple(self, prop, value, singleton_class):
        if type(value) in (tuple, list, np.ndarray):
            tvalue = tuple(singleton_class(v) for v in value)
            if len(tvalue) > 1 and len(self) > 1 and len(tvalue) != len(self):
                raise ValueError(f"Length of {prop} must be 1 or {len(self)}")
        else:
            value = singleton_class(value)
            tvalue = (value,)
        self.__setattr__(f'_{prop}', tvalue)