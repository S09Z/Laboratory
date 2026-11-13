import signal
from pathlib import Path

import pytest

from colab import ckpt, runtime, train_runner


def make_ckpt(d: Path, step: int, parts=("G", "D", "WD"), payload="ok"):
    for p in parts:
        (d / f"{p}_{step}.pth").write_text(payload)


def loader(path):
    if Path(path).read_text() != "ok":
        raise RuntimeError("corrupt")


def test_pick_batch_size_t4():
    assert runtime.pick_batch_size(15.0) == 4
    assert runtime.pick_batch_size(8) == 1


def test_resume_skips_corrupt_latest(tmp_path):
    make_ckpt(tmp_path, 1000)
    make_ckpt(tmp_path, 1500, payload="bad")
    assert ckpt.find_resumable_checkpoint(tmp_path, loader=loader) == 1000
    assert (tmp_path / "corrupt" / "G_1500.pth").exists()
    assert not (tmp_path / "G_1500.pth").exists()


def test_resume_requires_all_three(tmp_path):
    make_ckpt(tmp_path, 1000)
    make_ckpt(tmp_path, 2000, parts=("G",))
    assert ckpt.find_resumable_checkpoint(tmp_path, loader=loader) == 1000


def test_resume_none_when_empty(tmp_path):
    assert ckpt.find_resumable_checkpoint(tmp_path, loader=loader) is None


def test_prune_keeps_latest_three(tmp_path):
    for s in (500, 1000, 1500, 2000, 2500):
        make_ckpt(tmp_path, s)
    removed = ckpt.prune_checkpoints(tmp_path, keep=3)
    assert ckpt.complete_steps(tmp_path) == [1500, 2000, 2500]
    assert len(removed) == 6


def test_ensure_gpu_raises_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="No GPU"):
        runtime.ensure_gpu()


def test_data_hash_stable(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"12")
    (tmp_path / "b.wav").write_bytes(b"345")
    h1 = runtime.data_hash(tmp_path)
    assert h1 == runtime.data_hash(tmp_path)
    (tmp_path / "c.wav").write_bytes(b"6")
    assert runtime.data_hash(tmp_path) != h1


class FakeProc:
    def __init__(self, returncode_on_exit=None):
        self.returncode = None
        self.signals = []
        self._exit = returncode_on_exit

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = -2

    def wait(self, timeout=None):
        return self.returncode


def test_run_training_budget_waits_for_next_ckpt(tmp_path):
    make_ckpt(tmp_path, 500)
    proc = FakeProc()
    t = {"now": 0.0}

    def sleep(s):
        t["now"] += s
        if t["now"] >= 150 and not (tmp_path / "G_1000.pth").exists():
            assert proc.signals == []  # no SIGINT before the new checkpoint exists
            make_ckpt(tmp_path, 1000)

    res = train_runner.run_training(["x"], tmp_path, budget_s=100, poll_s=30,
                                    popen=lambda *a, **k: proc, clock=lambda: t["now"],
                                    sleep=sleep, log_path=tmp_path / "log.txt", loader=loader)
    assert res == "budget_stop"
    assert proc.signals == [signal.SIGINT]


def test_run_training_reports_oom(tmp_path):
    log = tmp_path / "log.txt"
    proc = FakeProc()

    def popen(*a, **k):
        log.write_text("RuntimeError: CUDA out of memory. Tried to allocate")
        proc.returncode = 1
        return proc

    res = train_runner.run_training(["x"], tmp_path, budget_s=100, popen=popen,
                                    clock=lambda: 0.0, sleep=lambda s: None, log_path=log, loader=loader)
    assert res == "oom"


def test_run_training_finished(tmp_path):
    proc = FakeProc()
    proc.returncode = 0
    res = train_runner.run_training(["x"], tmp_path, budget_s=100, popen=lambda *a, **k: proc,
                                    clock=lambda: 0.0, sleep=lambda s: None,
                                    log_path=tmp_path / "l.txt", loader=loader)
    assert res == "finished"


# ---- final-review fixes ----
import yaml


def test_write_sbv2_config_yml_sets_model_and_keep(tmp_path):
    (tmp_path / "default_config.yml").write_text(
        "model_name: \"model_name\"\ntrain_ms:\n  keep_ckpts: 1\n  model: models\n", encoding="utf-8")
    runtime.write_sbv2_config_yml(tmp_path, "amitaro_v1_E2", tmp_path / "Data" / "amitaro_v1_E2", keep_ckpts=3)
    cfg = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    assert cfg["model_name"] == "amitaro_v1_E2"
    assert cfg["dataset_path"] == str(tmp_path / "Data" / "amitaro_v1_E2")
    assert cfg["train_ms"]["keep_ckpts"] == 3 and cfg["train_ms"]["model"] == "models"


def test_copy_pretrained_init(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    for p in ("G", "D", "WD"):
        (src / f"{p}_0.safetensors").write_text("w")
    (src / "G_500.pth").write_text("ok")
    assert runtime.copy_pretrained_init(src, dst) == 3
    assert sorted(f.name for f in dst.iterdir()) == ["D_0.safetensors", "G_0.safetensors", "WD_0.safetensors"]


def test_resume_moves_newer_partial_files_aside(tmp_path):
    make_ckpt(tmp_path, 500)
    (tmp_path / "G_1000.pth").write_text("half")   # session died while saving step 1000
    assert ckpt.find_resumable_checkpoint(tmp_path, loader=loader) == 500
    assert not (tmp_path / "G_1000.pth").exists()
    assert (tmp_path / "corrupt" / "G_1000.pth").exists()


def test_resume_none_moves_all_partials(tmp_path):
    (tmp_path / "G_1000.pth").write_text("half")
    assert ckpt.find_resumable_checkpoint(tmp_path, loader=loader) is None
    assert not (tmp_path / "G_1000.pth").exists()


def test_prune_safetensors_keeps_latest(tmp_path):
    for s in (500, 1000, 1500, 2000):
        (tmp_path / f"m_e1_s{s}.safetensors").write_text("x")
    ckpt.prune_safetensors(tmp_path, keep=2)
    assert sorted(f.name for f in tmp_path.glob("*.safetensors")) == ["m_e1_s1500.safetensors", "m_e1_s2000.safetensors"]


def test_budget_stop_waits_for_inference_safetensors(tmp_path):
    models, assets = tmp_path / "models", tmp_path / "assets"
    models.mkdir(); assets.mkdir()
    make_ckpt(models, 500)
    proc = FakeProc()
    t = {"now": 0.0}

    def sleep(s):
        t["now"] += s
        if t["now"] >= 150 and not (models / "G_1000.pth").exists():
            make_ckpt(models, 1000)
        elif (models / "G_1000.pth").exists() and t["now"] >= 300:
            assert proc.signals == []   # not before the safetensors exists
            (assets / "m_e1_s1000.safetensors").write_text("x")

    res = train_runner.run_training(["x"], models, budget_s=100, poll_s=30, popen=lambda *a, **k: proc,
                                    clock=lambda: t["now"], sleep=sleep, log_path=tmp_path / "l.txt",
                                    loader=loader, assets_dir=assets, grace_s=30)
    assert res == "budget_stop" and proc.signals == [signal.SIGINT]
    assert t["now"] >= 330


def test_old_oom_in_log_does_not_count(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("earlier session: CUDA out of memory\n")
    proc = FakeProc()

    def popen(*a, **k):
        with open(log, "a") as f:
            f.write("KeyError: something else\n")
        proc.returncode = 1
        return proc

    res = train_runner.run_training(["x"], tmp_path, budget_s=100, popen=popen, clock=lambda: 0.0,
                                    sleep=lambda s: None, log_path=log, loader=loader)
    assert res == "failed"


# ---- second review round ----
def test_ensure_pretrained_init_copies_from_sbv2_and_counts(tmp_path):
    pre = tmp_path / "pretrained_jp_extra"
    pre.mkdir()
    for p in ("G", "D", "WD"):
        (pre / f"{p}_0.safetensors").write_text("w")
    models = tmp_path / "models"
    assert runtime.ensure_pretrained_init(pre, models) == 3
    assert sorted(f.name for f in models.iterdir()) == ["D_0.safetensors", "G_0.safetensors", "WD_0.safetensors"]


def test_ensure_pretrained_init_raises_when_source_missing(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="pretrained"):
        runtime.ensure_pretrained_init(tmp_path / "nope", tmp_path / "models")


def test_interrupt_stops_trainer_subprocess(tmp_path):
    proc = FakeProc()

    def sleep(s):
        raise KeyboardInterrupt  # user presses Stop in Colab

    import pytest
    with pytest.raises(KeyboardInterrupt):
        train_runner.run_training(["x"], tmp_path, budget_s=100, popen=lambda *a, **k: proc,
                                  clock=lambda: 0.0, sleep=sleep, log_path=tmp_path / "l.txt", loader=loader)
    assert proc.signals == [signal.SIGINT]


def test_on_prune_callback_called_after_pruning(tmp_path):
    make_ckpt(tmp_path, 500)
    proc = FakeProc()
    calls = []
    t = {"n": 0}

    def sleep(s):
        t["n"] += 1
        if t["n"] == 1:
            make_ckpt(tmp_path, 1000)
        if t["n"] == 2:
            proc.returncode = 0

    train_runner.run_training(["x"], tmp_path, budget_s=1e9, popen=lambda *a, **k: proc, clock=lambda: 0.0,
                              sleep=sleep, log_path=tmp_path / "l.txt", loader=loader,
                              on_prune=lambda: calls.append(1))
    assert calls == [1]
