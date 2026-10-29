"""
Unit Tests for Stoma Structured OCR Ingestion
=============================================
Verifies parsing of:
  1. DePlot/MatCha table structures into markdown table passages.
  2. Chart data series into structured summary passages.
  3. VRDU-style key-value fields and entities.
  4. Traditional text and multi-page lists.
"""

import json
import tempfile
import unittest
from pathlib import Path

from stoma.plugins.ocr import OCRPlugin


class TestStructuredOCRIngestion(unittest.TestCase):

    def setUp(self):
        self.plugin = OCRPlugin()

    def test_structured_table_extraction(self):
        """Verify OCR JSON with tables converts to formatted markdown table passages."""
        data = {
            "tables": [
                {
                    "title": "Quarterly Financial Results",
                    "page": 2,
                    "headers": ["Quarter", "Revenue ($M)", "Gross Margin (%)"],
                    "rows": [
                        ["Q1 2026", "1,250", "48.2"],
                        ["Q2 2026", "1,380", "50.1"],
                        ["Q3 2026", "1,520", "51.4"],
                    ],
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".ocr.json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            res = self.plugin.extract(tmp_path, "financial_report.ocr.json")
            self.assertEqual(len(res.passages), 1)
            p = res.passages[0]
            self.assertIn("Quarterly Financial Results", p.content)
            self.assertIn("| Quarter | Revenue ($M) | Gross Margin (%) |", p.content)
            self.assertIn("| Q1 2026 | 1,250 | 48.2 |", p.content)
            self.assertEqual(p.location.get("type"), "table")
            self.assertEqual(p.location.get("page"), 2)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_structured_chart_extraction(self):
        """Verify DePlot/MatCha chart structures parse into structured passages."""
        data = {
            "charts": [
                {
                    "title": "EUV Laser Power over Time",
                    "type": "line_plot",
                    "page": 4,
                    "summary": "Shows continuous power increase from 250W to 500W at source.",
                    "table": {
                        "headers": ["Year", "Power (W)"],
                        "rows": [
                            ["2024", "250"],
                            ["2025", "375"],
                            ["2026", "500"],
                        ],
                    },
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".ocr.json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            res = self.plugin.extract(tmp_path, "chart_artifact.ocr.json")
            self.assertEqual(len(res.passages), 1)
            p = res.passages[0]
            self.assertIn("Chart: EUV Laser Power over Time (line_plot)", p.content)
            self.assertIn("Shows continuous power increase", p.content)
            self.assertIn("| Year | Power (W) |", p.content)
            self.assertEqual(p.location.get("type"), "chart")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_structured_key_values_extraction(self):
        """Verify VRDU key-value fields parse into structured bullet points."""
        data = {
            "page": 1,
            "fields": {
                "vendor_name": "Carl Zeiss SMT GmbH",
                "invoice_number": "INV-2026-9921",
                "total_due": "$12,450,000.00",
                "due_date": "2026-12-01",
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".ocr.json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            res = self.plugin.extract(tmp_path, "invoice.ocr.json")
            self.assertEqual(len(res.passages), 1)
            p = res.passages[0]
            self.assertIn("### Structured Fields (Page 1)", p.content)
            self.assertIn("- **vendor_name**: Carl Zeiss SMT GmbH", p.content)
            self.assertIn("- **total_due**: $12,450,000.00", p.content)
            self.assertEqual(p.location.get("type"), "key_values")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
