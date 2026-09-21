"""Regression tests for package.py's pure notebook-generation logic.

No network/GPU: these only check that generated notebook JSON is
well-formed and that every registered stage (including the ISSUES.md #15
option-1 paraphraser comparison probe) round-trips through ast.parse and
json.loads without executing any cell.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from package import NOTEBOOK_STAGE_CELLS, _strip_local_imports, notebook_source


class NotebookSourceTests(unittest.TestCase):
    def test_every_registered_stage_produces_valid_notebook_json(self) -> None:
        bundle = b"not-a-real-zip-but-bytes-are-opaque-to-notebook_source"
        for stage in NOTEBOOK_STAGE_CELLS:
            source = notebook_source(bundle, stage)
            notebook = json.loads(source)
            self.assertGreaterEqual(len(notebook["cells"]), 2)
            for cell in notebook["cells"]:
                self.assertEqual(cell["cell_type"], "code")

    def test_every_cell_source_is_syntactically_valid_python(self) -> None:
        for stage, cells in NOTEBOOK_STAGE_CELLS.items():
            resolved = cells(Path(__file__).parent) if callable(cells) else cells
            for body in resolved:
                with self.subTest(stage=stage):
                    ast.parse(body)

    def test_unregistered_stage_raises(self) -> None:
        with self.assertRaises(ValueError):
            notebook_source(b"x", "not-a-real-stage")

    def test_paraphrase_probe_3b_cells_reference_pinned_candidate_revision(self) -> None:
        # Regression: this probe must always run the exact pinned candidate
        # (ISSUES.md #15 option 1), never an unpinned "main" revision.
        cells = NOTEBOOK_STAGE_CELLS["paraphrase_probe_3b"]
        combined = "\n".join(cells)
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", combined)
        self.assertIn("aa8e72537993ba99e69dfaafa59ed015b17504d1", combined)

    def test_full_corpus_generation_has_no_zip_or_base64_embedding(self) -> None:
        # Regression for ISSUES.md #23 / the "stop zipping and pushing"
        # requirement: this stage must be fully readable/editable in the
        # Kaggle notebook UI, not an opaque compressed source blob.
        source = notebook_source(b"unused", "full_corpus_generation")
        notebook = json.loads(source)
        combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertNotIn("base64", combined)
        self.assertNotIn("zipfile", combined)
        self.assertGreaterEqual(len(notebook["cells"]), 8)

    def test_full_corpus_generation_inlines_real_tested_source(self) -> None:
        source = notebook_source(b"unused", "full_corpus_generation")
        notebook = json.loads(source)
        combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("def process_catalog_records_incrementally", combined)
        self.assertIn("def write_sharded_records", combined)
        self.assertIn("def resolve_full_catalog_metadata", combined)
        self.assertIn("class Florence2Captioner", combined)
        self.assertNotIn("from crosswalk import", combined)
        self.assertNotIn("from caption import", combined)
        self.assertNotIn("from corpus import", combined)

    def test_strip_local_imports_removes_cross_module_imports_only(self) -> None:
        source = (
            "import json\n"
            "from crosswalk import CATALOG_SIZE\n"
            "from caption import (\n"
            "    Florence2Captioner,\n"
            "    QwenParaphraser,\n"
            ")\n"
            "x = 1\n"
        )
        stripped = _strip_local_imports(source)
        self.assertIn("import json", stripped)
        self.assertIn("x = 1", stripped)
        self.assertNotIn("from crosswalk", stripped)
        self.assertNotIn("from caption", stripped)
        ast.parse(stripped)


if __name__ == "__main__":
    unittest.main()
