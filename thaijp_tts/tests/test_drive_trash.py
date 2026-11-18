from colab.drive_trash import CHECKPOINT_NAME, purge_trashed_checkpoints


class FakeFiles:
    def __init__(self, items):
        self.items, self.deleted, self.queries = items, [], []

    def list(self, q, fields, pageSize, pageToken=None):
        self.queries.append(q)
        return type("R", (), {"execute": lambda s: {"files": self.items}})()

    def delete(self, fileId):
        self.deleted.append(fileId)
        return type("R", (), {"execute": lambda s: None})()


class FakeService:
    def __init__(self, items):
        self._f = FakeFiles(items)

    def files(self):
        return self._f


def test_purges_only_checkpoint_files_from_given_folders():
    svc = FakeService([
        {"id": "1", "name": "G_500.pth", "parents": ["models"]},
        {"id": "2", "name": "amitaro_e1_s500.safetensors", "parents": ["assets"]},
        {"id": "3", "name": "my_holiday_photo.jpg", "parents": ["models"]},
        {"id": "4", "name": "G_500.pth", "parents": ["someone_else"]},
    ])
    n = purge_trashed_checkpoints(svc, {"models", "assets"})
    assert n == 2 and svc.files().deleted == ["1", "2"]
    assert all("trashed = true" in q for q in svc.files().queries)


def test_checkpoint_name_pattern():
    for ok in ("G_500.pth", "D_1000.pth", "WD_0.pth", "x_e3_s1500.safetensors"):
        assert CHECKPOINT_NAME.match(ok)
    for bad in ("G_0.safetensors", "notes.txt", "G_500.pth.bak"):
        assert not CHECKPOINT_NAME.match(bad)
