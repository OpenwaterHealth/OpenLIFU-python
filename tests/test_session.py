from __future__ import annotations

import pytest

from openlifu.db.session import Session
from openlifu.geo import Point


def test_duplicate_target_ids_raises():
    target1 = Point(id="T1", position=[0, 0, 0])
    target2 = Point(id="T1", position=[1, 1, 1])  # Duplicate ID

    with pytest.raises(ValueError, match="Duplicate target IDs found"):
        Session(targets=[target1, target2])
