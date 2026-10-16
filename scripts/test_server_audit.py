"""Small stdlib regression checks for leakage counts and non-overwrite behavior."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from audit_edge_data import PACKET_FIELDS, audit
from check_edge_server import split_check


class ServerAuditChecks(unittest.TestCase):
    def test_label_independent_overlap_and_local_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            fields = ["flow_feature", "Label"] + ["udps." + k for k in PACKET_FIELDS]
            base = {k: "[0]" for k in fields}
            base.update({"flow_feature": "1", "Label": "0", "udps.payload_data": "['00']"})
            train, test = folder / "train.csv", folder / "test.csv"
            for path, rows in ((train, [base, base]),
                               (test, [base, dict(base, Label="1")])):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            output = folder / "audit.json"
            audit([train, test], output)
            result = json.loads(output.read_text())
            self.assertEqual(result["sources"][0]["within_file_duplicate_rows"], 1)
            self.assertEqual(result["sources"][1]["within_file_duplicate_rows"], 0)
            self.assertEqual(result["cross_file_duplicate_rows"], 1)
            self.assertEqual(result["cross_file_duplicate_features"], 2)
            self.assertEqual(result["conflicting_label_rows"], 1)
            self.assertTrue(all(s["invalid_rows"] == 0 for s in result["sources"]))
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                audit([train, test], output)
            self.assertEqual(output.read_bytes(), original)

    def test_old_split_reports_overlap_and_invalid_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_text(json.dumps({"total_graphs": 3, "data_path": str(Path(folder) / "missing.pt"),
                "split_indices": {"train": [0, 0], "val": [0], "test": [4]}}))
            result = split_check(path)
            self.assertEqual(result["cross_split_index_overlap"]["train/val"], 1)
            self.assertEqual(result["repeated_indices"]["train"], 1)
            self.assertEqual(result["invalid_index_count"], 1)
            self.assertFalse(result["recorded_data_paths"]["data_path"]["exists_on_this_host"])


if __name__ == "__main__":
    unittest.main()
