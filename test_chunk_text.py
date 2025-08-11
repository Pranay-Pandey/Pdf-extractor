import unittest
from main import chunk_text

class TestChunkText(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text(self):
        text = "Hello world!"
        self.assertEqual(chunk_text(text, chunk_size=1000), [text])

    def test_exact_chunk_size(self):
        text = "a" * 1000
        self.assertEqual(chunk_text(text, chunk_size=1000), [text])

    def test_multiple_chunks(self):
        text = "a" * 2500
        expected = ["a" * 1000, "a" * 1000, "a" * 500]
        self.assertEqual(chunk_text(text, chunk_size=1000), expected)

    def test_custom_chunk_size(self):
        text = "abcdefg"
        self.assertEqual(chunk_text(text, chunk_size=2), ["ab", "cd", "ef", "g"])

if __name__ == "__main__":
    unittest.main()
