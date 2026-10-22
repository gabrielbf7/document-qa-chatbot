import pytest

from src import uploads


@pytest.mark.parametrize(
    "name, expected",
    [
        ("notes.txt", "notes.txt"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\secret.md", "secret.md"),
        ("my file (1).pdf", "my file _1_.pdf"),
        ("report.exe", None),
        ("noextension", None),
        ("..txt", None),
        ("", None),
    ],
)
def test_sanitize_filename(name, expected):
    assert uploads.sanitize_filename(name) == expected


def test_save_upload_writes_only_inside_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "ROOT_DIR", tmp_path)
    (tmp_path / "knowledge_base").mkdir()
    (tmp_path / "knowledge_base" / "faq.md").write_text("x")

    saved = uploads.save_upload("../../evil.txt", b"hello")
    assert saved == tmp_path / "uploads" / "evil.txt"
    assert saved.read_bytes() == b"hello"

    renamed = uploads.save_upload("faq.md", b"y")
    assert renamed.name == "upload_faq.md"
    assert uploads.save_upload("virus.exe", b"z") is None
