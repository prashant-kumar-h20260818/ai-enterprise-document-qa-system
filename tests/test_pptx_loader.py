from src.loaders import _shape_position


class FakeShape:
    def __init__(self, left=None, top=None):
        self.left = left
        self.top = top


def test_shape_position_handles_missing_coordinates():
    assert _shape_position(FakeShape(None, None)) == "position unavailable"
    assert _shape_position(FakeShape(None, 100)) == "position unavailable"
    assert _shape_position(FakeShape(100, None)) == "position unavailable"


def test_shape_position_formats_valid_coordinates():
    assert _shape_position(FakeShape(100, 200)) == "x=100, y=200"
