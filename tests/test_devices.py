from thaijp_accent.devices import find_drive_lab, pick_device


def test_pick_device_prefers_mps_when_probe_succeeds():
    assert pick_device(available={"mps": True}, probe=lambda d: None) == "mps"


def test_pick_device_falls_back_to_cpu_when_mps_probe_fails():
    def probe(d):
        if d == "mps":
            raise NotImplementedError("aten::op not implemented for MPS")
    assert pick_device(available={"mps": True}, probe=probe) == "cpu"


def test_pick_device_cpu_without_mps():
    assert pick_device(available={"mps": False}, probe=lambda d: None) == "cpu"


def test_find_drive_lab_in_cloudstorage(tmp_path):
    lab = tmp_path / "Library" / "CloudStorage" / "GoogleDrive-a@b.com" / "My Drive" / "Laboratory"
    lab.mkdir(parents=True)
    assert find_drive_lab(home=tmp_path) == lab


def test_find_drive_lab_thai_mydrive_name(tmp_path):
    lab = tmp_path / "Library" / "CloudStorage" / "GoogleDrive-a@b.com" / "ไดรฟ์ของฉัน" / "Laboratory"
    lab.mkdir(parents=True)
    assert find_drive_lab(home=tmp_path) == lab


def test_find_drive_lab_none(tmp_path):
    assert find_drive_lab(home=tmp_path) is None
