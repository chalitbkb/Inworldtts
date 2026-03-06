"""Unit tests for Thai text normalization pipeline.

Tests the _normalize_thai_text function and its helper _convert_thai_numerals,
covering all 9 stages of the Thai normalization pipeline.

Note: These tests require the `pythainlp` package to be installed.
      If not installed, tests involving _normalize_thai_text will be skipped.
"""

import re
import unittest

from tts.data.text_normalization import (
    _convert_thai_numerals,
    _THAI_ABBREVIATIONS,
    _THAI_DIGIT_NAMES,
    _THAI_MONTHS,
    _THAI_NUMERAL_MAP,
    _THAI_UNITS,
)

# Try importing the main function — skip tests if pythainlp is missing.
try:
    from pythainlp.util import num_to_thaiword
    _HAS_PYTHAINLP = True
except ImportError:
    _HAS_PYTHAINLP = False

# Conditional import of _normalize_thai_text.
from tts.data.text_normalization import _normalize_thai_text


class TestConvertThaiNumerals(unittest.TestCase):
    """Stage 1: Thai numeral conversion (does NOT require pythainlp)."""

    def test_single_digits(self):
        self.assertEqual(_convert_thai_numerals("๐"), "0")
        self.assertEqual(_convert_thai_numerals("๑"), "1")
        self.assertEqual(_convert_thai_numerals("๙"), "9")

    def test_multi_digit(self):
        self.assertEqual(_convert_thai_numerals("๑๒๓"), "123")
        self.assertEqual(_convert_thai_numerals("๒๕๖๘"), "2568")

    def test_mixed_text(self):
        self.assertEqual(
            _convert_thai_numerals("ปี ๒๕๖๘ เดือน ๓"),
            "ปี 2568 เดือน 3",
        )

    def test_no_thai_numerals(self):
        self.assertEqual(_convert_thai_numerals("hello 123"), "hello 123")

    def test_empty_string(self):
        self.assertEqual(_convert_thai_numerals(""), "")


class TestConstantCompleteness(unittest.TestCase):
    """Verify that constant dictionaries are complete."""

    def test_thai_numeral_map_has_all_digits(self):
        """Must have all 10 Thai numerals ๐-๙."""
        self.assertEqual(len(_THAI_NUMERAL_MAP), 10)
        for thai_char in "๐๑๒๓๔๕๖๗๘๙":
            self.assertIn(thai_char, _THAI_NUMERAL_MAP)

    def test_thai_digit_names_has_all_digits(self):
        """Must have all 10 Arabic digits 0-9."""
        self.assertEqual(len(_THAI_DIGIT_NAMES), 10)
        for digit in "0123456789":
            self.assertIn(digit, _THAI_DIGIT_NAMES)

    def test_thai_months_has_12_months(self):
        """Must have 12 months + 1 empty placeholder at index 0."""
        self.assertEqual(len(_THAI_MONTHS), 13)
        self.assertEqual(_THAI_MONTHS[0], "")
        self.assertEqual(_THAI_MONTHS[1], "มกราคม")
        self.assertEqual(_THAI_MONTHS[12], "ธันวาคม")

    def test_thai_units_not_empty(self):
        self.assertGreater(len(_THAI_UNITS), 0)

    def test_thai_abbreviations_not_empty(self):
        self.assertGreater(len(_THAI_ABBREVIATIONS), 0)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextNumbers(unittest.TestCase):
    """Stage 9: Number-to-word conversion."""

    def test_integer(self):
        result = _normalize_thai_text("100")
        self.assertEqual(result, "หนึ่งร้อย")

    def test_negative_number(self):
        result = _normalize_thai_text("-5")
        self.assertEqual(result, "ลบห้า")

    def test_decimal(self):
        result = _normalize_thai_text("3.14")
        self.assertIn("สาม", result)  # "สามจุดหนึ่งสี่"

    def test_number_in_context(self):
        result = _normalize_thai_text("ราคา 100 บาท")
        self.assertIn("หนึ่งร้อย", result)
        self.assertNotIn("100", result)

    def test_no_numbers(self):
        result = _normalize_thai_text("สวัสดีครับ")
        self.assertEqual(result, "สวัสดีครับ")

    def test_empty_string(self):
        result = _normalize_thai_text("")
        self.assertEqual(result, "")


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextTime(unittest.TestCase):
    """Stage 3: Time normalization."""

    def test_time_colon(self):
        result = _normalize_thai_text("14:30")
        self.assertIn("นาฬิกา", result)
        self.assertIn("นาที", result)
        self.assertNotIn("14", result)

    def test_time_dot(self):
        result = _normalize_thai_text("8.00 น.")
        self.assertIn("นาฬิกา", result)
        self.assertNotIn("8", result)

    def test_time_zero_minutes(self):
        result = _normalize_thai_text("9:00")
        self.assertIn("นาฬิกา", result)
        self.assertNotIn("นาที", result)  # 0 minutes = no "นาที"


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextPhone(unittest.TestCase):
    """Stage 5: Phone number handling."""

    def test_mobile_no_dashes(self):
        result = _normalize_thai_text("0812345678")
        self.assertIn("ศูนย์", result)
        self.assertIn("แปด", result)
        # Should NOT contain huge number words
        self.assertNotIn("ล้าน", result)

    def test_mobile_with_dashes(self):
        result = _normalize_thai_text("081-234-5678")
        self.assertIn("ศูนย์", result)
        self.assertNotIn("-", result)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextDate(unittest.TestCase):
    """Stage 4: Date normalization."""

    def test_thai_date_format(self):
        result = _normalize_thai_text("1/1/2568")
        self.assertIn("วันที่", result)
        self.assertIn("มกราคม", result)
        self.assertIn("พุทธศักราช", result)

    def test_iso_date_format(self):
        result = _normalize_thai_text("2025-03-06")
        self.assertIn("วันที่", result)
        self.assertIn("มีนาคม", result)
        self.assertIn("คริสต์ศักราช", result)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextUnits(unittest.TestCase):
    """Stage 7: Unit/symbol expansion."""

    def test_celsius(self):
        result = _normalize_thai_text("35°C")
        self.assertIn("องศาเซลเซียส", result)
        self.assertNotIn("°C", result)

    def test_percent(self):
        result = _normalize_thai_text("80%")
        self.assertIn("เปอร์เซ็นต์", result)
        self.assertNotIn("%", result)

    def test_kmh(self):
        result = _normalize_thai_text("120km/h")
        self.assertIn("กิโลเมตรต่อชั่วโมง", result)

    def test_baht_symbol(self):
        result = _normalize_thai_text("฿500")
        self.assertIn("บาท", result)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextAbbreviations(unittest.TestCase):
    """Stage 6: Abbreviation expansion."""

    def test_bangkok(self):
        result = _normalize_thai_text("กทม.")
        self.assertIn("กรุงเทพมหานคร", result)

    def test_hospital(self):
        result = _normalize_thai_text("รพ.")
        self.assertIn("โรงพยาบาล", result)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextThaiNumerals(unittest.TestCase):
    """Stage 1: Thai numeral conversion (within full pipeline)."""

    def test_thai_numerals_converted(self):
        result = _normalize_thai_text("ปี ๒๕๖๘")
        # The Thai year should be converted to words
        self.assertNotIn("๒", result)
        self.assertNotIn("2568", result)  # Also converted to words


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextMaiYamok(unittest.TestCase):
    """Stage 8: Mai Yamok (ๆ) handling."""

    def test_mai_yamok_removed(self):
        result = _normalize_thai_text("เด็กๆ")
        self.assertNotIn("ๆ", result)


@unittest.skipUnless(_HAS_PYTHAINLP, "pythainlp not installed")
class TestNormalizeThaiTextIntegration(unittest.TestCase):
    """Integration tests with mixed content."""

    def test_mixed_sentence(self):
        """Test a sentence with numbers, units, and Thai text."""
        result = _normalize_thai_text("อุณหภูมิวันนี้ 35°C")
        self.assertIn("องศาเซลเซียส", result)
        self.assertNotIn("35", result)
        self.assertNotIn("°C", result)

    def test_passthrough_no_special(self):
        """Plain Thai text should pass through unchanged."""
        text = "สวัสดีครับ วันนี้อากาศดี"
        result = _normalize_thai_text(text)
        self.assertIn("สวัสดี", result)


if __name__ == "__main__":
    unittest.main()
