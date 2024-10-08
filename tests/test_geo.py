import numpy as np

from openlifu.geo import Point


def test_point_from_dict():
    point = Point.from_dict({'position' : [10,20,30],})
    assert (point.position == np.array([10,20,30], dtype=float)).all()
