"""Tests for setup.py's Flint family (no GPU, no network, nothing installed): the entry is complete and pinned, the sizes it offers have
a SHA-256, and a config written for it reads back as Flint's.

    python -m unittest tools.test_setup_flint
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class FlintFamily(unittest.TestCase):
    fam = setup.OWN_PATH_FAMILIES["flint"]

    def test_the_entry_is_complete(self):
        f = self.fam
        for key in ("title", "by", "arch", "about", "hf", "file", "tag", "name", "models", "sha256", "kv_bytes_per_token"):
            self.assertIn(key, f, key)
        self.assertEqual(f["arch"], "flint")
        self.assertFalse(f["vision"])
        self.assertFalse(f["parallel"])
        self.assertFalse(f["mtp"])

    def test_every_size_has_a_pinned_file_with_a_size_and_a_sha256(self):
        for model in self.fam["models"]:
            name = self.fam["file"].format(q=model)
            size, sha = self.fam["sha256"][name]
            self.assertGreater(size, 1e9)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertRegex(self.fam["hf"], r"/unsloth/flint-GGUF/resolve/[0-9a-f]{40}/$")

    def test_the_file_is_in_the_pin_table_that_lets_it_be_downloaded(self):
        url = self.fam["hf"] + self.fam["file"].format(q="UD-Q4_K_XL")
        self.assertEqual(setup.file_pin(url), self.fam["sha256"]["flint-UD-Q4_K_XL.gguf"])

    def test_the_download_url_is_the_file_of_the_pinned_revision(self):
        url = self.fam["hf"] + self.fam["file"].format(q="UD-Q4_K_XL")
        self.assertTrue(url.endswith("/resolve/" + setup.HF_REVISIONS["unsloth/flint-GGUF"] + "/flint-UD-Q4_K_XL.gguf"))

    def test_it_is_offered_where_the_other_own_path_families_are(self):
        self.assertIn("flint", setup.PREVIEW_FAMILIES)
        self.assertTrue(setup.own_path_config({"arch": "flint"}))
        self.assertEqual(setup.own_path_size(self.fam, 15), "UD-Q4_K_XL")
        self.assertEqual(setup.own_path_size(self.fam, 4), "UD-Q4_K_XL")        # (the smallest when none fits)
        self.assertEqual(setup.own_path_context(self.fam, 16.0), 32768)

    def test_a_config_reads_back_as_flint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-flint-ud-q4_k_xl.json"
            p.write_text(json.dumps({"arch": "flint", "port": 8080, "args": ["--native", "/m/flint-UD-Q4_K_XL.gguf",
                                                                               "--max-context", "32768", "--prefill", "auto"]}))
            c = setup.choices_from_config(p)
            self.assertEqual((c["family"], c["model"], c["context"], c["vision"]), ("flint", "UD-Q4_K_XL", 32768, "none"))


if __name__ == "__main__":
    unittest.main()
