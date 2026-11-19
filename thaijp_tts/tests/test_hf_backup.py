from pathlib import Path

from colab import hf_backup


class FakeApi:
    """In-memory stand-in for huggingface_hub.HfApi + download."""

    def __init__(self, files=None, fail_upload_for=None):
        self.files: dict[str, bytes] = dict(files or {})
        self.created = []
        self.fail_upload_for = fail_upload_for

    def create_repo(self, repo_id, private=True, exist_ok=True, repo_type="model"):
        self.created.append((repo_id, private))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
        if self.fail_upload_for and self.fail_upload_for in path_in_repo:
            raise OSError("network down")
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        self.files[path_in_repo] = data

    def list_repo_files(self, repo_id, **kw):
        return sorted(self.files)

    def delete_file(self, path_in_repo, repo_id, **kw):
        del self.files[path_in_repo]

    def download(self, repo_id, filename, local_dir):
        dst = Path(local_dir) / filename   # like hf_hub_download(local_dir=...): keeps repo subfolders
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.files[filename])
        return str(dst)


def make_run(tmp_path, steps=(1000,)):
    models, assets = tmp_path / "models", tmp_path / "assets"
    models.mkdir()
    assets.mkdir()
    for s in steps:
        for p in ("G", "D", "WD"):
            (models / f"{p}_{s}.pth").write_bytes(f"{p}{s}".encode())
        (assets / f"run_e1_s{s}.safetensors").write_bytes(f"st{s}".encode())
    (assets / "config.json").write_text("{}")
    (assets / "style_vectors.npy").write_bytes(b"sv")
    return models, assets


def backup(tmp_path, api, models, assets, **kw):
    return hf_backup.HfBackup("u/repo", models, assets, api=api, download=api.download,
                              sleep=lambda s: None, **kw)


def test_push_uploads_newest_set_with_done_marker(tmp_path):
    models, assets = make_run(tmp_path, steps=(500, 1000))
    api = FakeApi()
    b = backup(tmp_path, api, models, assets)
    assert b.push() is True
    assert api.created == [("u/repo", True)]
    for f in ("G_1000.pth", "D_1000.pth", "WD_1000.pth", "run_e1_s1000.safetensors",
              "config.json", "style_vectors.npy", "DONE"):
        assert f"ckpt/1000/{f}" in api.files
    assert not any(k.startswith("ckpt/500/") for k in api.files)


def test_push_skips_when_nothing_new(tmp_path):
    models, assets = make_run(tmp_path)
    api = FakeApi()
    b = backup(tmp_path, api, models, assets)
    assert b.push() is True
    api.files.clear()
    assert b.push() is False and api.files == {}


def test_push_failure_is_swallowed_and_retried(tmp_path, capsys):
    models, assets = make_run(tmp_path)
    api = FakeApi(fail_upload_for="D_1000")
    b = backup(tmp_path, api, models, assets)
    assert b.push() is False            # must not raise: backup never stops training
    assert "ckpt/1000/DONE" not in api.files   # incomplete set is never marked done
    api.fail_upload_for = None
    assert b.push() is True
    assert "ckpt/1000/DONE" in api.files


def test_push_prunes_old_sets_keeping_two(tmp_path):
    models, assets = make_run(tmp_path, steps=(500,))
    api = FakeApi()
    b = backup(tmp_path, api, models, assets, keep=2)
    b.push()
    for s in (1000, 1500):
        for p in ("G", "D", "WD"):
            (models / f"{p}_{s}.pth").write_bytes(b"x")
        (assets / f"run_e1_s{s}.safetensors").write_bytes(b"x")
        b.push()
    steps = {k.split("/")[1] for k in api.files if k.startswith("ckpt/")}
    assert steps == {"1000", "1500"}


def test_push_waits_for_safetensors(tmp_path):
    models, assets = make_run(tmp_path)
    (assets / "run_e1_s1000.safetensors").unlink()
    api = FakeApi()
    slept = []

    def sleep(s):
        slept.append(s)
        if len(slept) == 2:  # trainer finishes writing the .safetensors
            (assets / "run_e1_s1000.safetensors").write_bytes(b"st")

    b = hf_backup.HfBackup("u/repo", models, assets, api=api, download=api.download, sleep=sleep)
    assert b.push() is True
    assert "ckpt/1000/run_e1_s1000.safetensors" in api.files


def test_restore_picks_newest_done_set_and_skips_partial(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    src_models, src_assets = make_run(src, steps=(1000,))
    api = FakeApi()
    backup(tmp_path, api, src_models, src_assets).push()
    api.files["ckpt/1500/G_1500.pth"] = b"partial"       # newer but no DONE → ignored
    dst_models, dst_assets = tmp_path / "m2", tmp_path / "a2"
    dst_models.mkdir()
    dst_assets.mkdir()
    b = hf_backup.HfBackup("u/repo", dst_models, dst_assets, api=api, download=api.download,
                           sleep=lambda s: None)
    assert b.restore() == 1000
    assert (dst_models / "G_1000.pth").read_bytes() == b"G1000"
    assert (dst_models / "WD_1000.pth").exists()
    assert (dst_assets / "run_e1_s1000.safetensors").exists()
    assert (dst_assets / "config.json").exists() and (dst_assets / "style_vectors.npy").exists()


def test_restore_keeps_local_checkpoints_and_handles_empty_repo(tmp_path):
    models, assets = make_run(tmp_path)
    api = FakeApi({"ckpt/9000/DONE": b"ok", "ckpt/9000/G_9000.pth": b"x"})
    assert backup(tmp_path, api, models, assets).restore() is None   # local set wins
    m2, a2 = tmp_path / "m2", tmp_path / "a2"
    m2.mkdir()
    a2.mkdir()
    assert hf_backup.HfBackup("u/repo", m2, a2, api=FakeApi(), download=FakeApi().download,
                              sleep=lambda s: None).restore() is None


def test_make_backup_none_without_token(capsys):
    assert hf_backup.make_backup("", Path("m"), Path("a"), token=None) is None
    assert "HF_TOKEN" in capsys.readouterr().out


def test_make_backup_derives_repo_from_whoami(tmp_path):
    class Api(FakeApi):
        def whoami(self):
            return {"name": "alice"}

    b = hf_backup.make_backup("", tmp_path, tmp_path, token="t", api=Api())
    assert b.repo_id == "alice/thaijp-tts-ckpt"
    b2 = hf_backup.make_backup("bob/x", tmp_path, tmp_path, token="t", api=Api())
    assert b2.repo_id == "bob/x"


def test_make_backup_passes_download_to_real_client_path(tmp_path, monkeypatch):
    """Regression: repo derived via whoami must not drop the download function."""
    import sys
    import types

    class RealishApi(FakeApi):
        def __init__(self, token=None):
            super().__init__()

        def whoami(self):
            return {"name": "alice"}

    fake_hub = types.SimpleNamespace(HfApi=RealishApi, hf_hub_download=lambda **k: "p")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    b = hf_backup.make_backup("", tmp_path, tmp_path, token="t")
    assert b.repo_id == "alice/thaijp-tts-ckpt" and callable(b.download)
