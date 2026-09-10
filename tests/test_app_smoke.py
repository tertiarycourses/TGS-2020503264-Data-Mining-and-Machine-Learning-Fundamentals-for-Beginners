from pathlib import Path

from streamlit.testing.v1 import AppTest

from workflow_catalog import WORKFLOW_GUIDES


ROOT = Path(__file__).resolve().parents[1]


def test_all_method_explainers_render_without_data():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=10).run()
    assert not app.exception

    for method in WORKFLOW_GUIDES:
        app.selectbox[0].select(method).run()
        assert not app.exception
        assert app.selectbox[0].value == method
