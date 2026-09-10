from pathlib import Path

from workflow_catalog import WORKFLOW_GUIDES


ROOT = Path(__file__).resolve().parents[1]


def test_every_lab_is_mapped_once_and_assets_exist():
    references = [lab for guide in WORKFLOW_GUIDES.values() for lab in guide.labs]
    assert sorted(lab.number for lab in references) == [f"{number:02d}" for number in range(1, 16)]

    for lab in references:
        folder = ROOT / "labs" / lab.folder
        assert (folder / "README.md").is_file()
        for filename in (name.strip() for name in lab.data.split("+")):
            assert (folder / filename).is_file()


def test_every_method_has_beginner_guidance():
    assert len(WORKFLOW_GUIDES) == 10
    for guide in WORKFLOW_GUIDES.values():
        assert len(guide.summary) >= 80
        assert len(guide.steps) == 3
        assert guide.use_when and guide.mechanism and guide.evidence and guide.caution
