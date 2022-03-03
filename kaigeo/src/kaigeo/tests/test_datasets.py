from kaigeo import datasets


def test_load_session1():
    session = datasets.load_session1()
    assert session.target_images.shape[0] == 66
