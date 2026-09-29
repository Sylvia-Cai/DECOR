from pathlib import Path
import tempfile
import unittest

from project_paths import OPEN_DECEPTION_RAW_DIR
from scripts.parse_open_deception import (
    EVALUATION_LABELS, _extract_dialogue_label, collect_transcripts, parse_transcript,
)


class OpenDeceptionParserTests(unittest.TestCase):
    def test_english_dataset_has_146_complete_dialogues(self):
        paths = [p for p in collect_transcripts(OPEN_DECEPTION_RAW_DIR)
                 if _extract_dialogue_label(p) in EVALUATION_LABELS]
        self.assertEqual(len(paths), 146)
        for path in paths:
            with self.subTest(path=str(path)):
                parsed = parse_transcript(path, {})
                expected = sum(line.startswith('AI (Speech):')
                               for line in path.read_text(encoding='utf-8').splitlines())
                self.assertGreater(expected, 0)
                self.assertEqual(len(parsed['rounds']), expected)

    def test_multiline_thought_and_incomplete_round(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dialogue.txt'
            path.write_text('user: incomplete\n[AI (Thought): stale]\n'
                            'user: complete\n[AI (Thought): first\nsecond]\n'
                            'AI (Speech): answer\ncontinued\n', encoding='utf-8')
            rounds = parse_transcript(path, {})['rounds']
            self.assertEqual(rounds, [{'round_id': 1, 'user': 'complete',
                                      'ai_thought': 'first\nsecond',
                                      'ai_speech': 'answer\ncontinued'}])


if __name__ == '__main__':
    unittest.main()
