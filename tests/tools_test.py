import poly_pitch_net as ppn


def test_get_project_root():
    root = ppn.tools.misc.get_project_root()

    assert root.stem == 'poly_pitch_net'
