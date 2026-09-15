import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from src.surprisal import encode_regions, score_tokens, score_story
from src.data import read_regions, validate_surprisals, build_analysis, write_tsv


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=5)

    def __call__(self, input_ids, use_cache=False):
        # The expected next token is current token plus one, mod vocabulary.
        logits = torch.zeros((*input_ids.shape, 11))
        logits.scatter_(2, ((input_ids + 1) % 11).unsqueeze(-1), 4)
        return SimpleNamespace(logits=logits)


class CausalTests(unittest.TestCase):
    def test_shift_and_window_coverage(self):
        scores = score_tokens(list(range(11)) + [0, 1], FakeModel(), 'cpu', stride=2)
        expected = math.log2(1 + 10 * math.exp(-4))
        self.assertTrue(math.isnan(scores[0]))
        for score in scores[1:]:
            self.assertAlmostEqual(score, expected, places=6)
        with self.assertRaises(ValueError):
            score_tokens([1, 2], FakeModel(), 'cpu', stride=5)

    def test_validation_rejects_partial_first_word(self):
        reference = [dict(item=1, zone=1, word='Hello')]
        with self.assertRaises(ValueError):
            validate_surprisals([dict(reference[0], surprisal=1)], reference)


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real cached GPT-2 BPE, without network access.
        cls.tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)

    def test_punctuation_and_multibpe(self):
        words = ['The', 'extraordinarily', 'long-bearded', 'butterfly', "didn't", 'disappear.']
        ids, owners = encode_regions(words, self.tokenizer)
        self.assertEqual(ids, self.tokenizer.encode(' '.join(words), add_special_tokens=False))
        self.assertGreater(owners.count(2), 1)
        for i, word in enumerate(words):
            pieces = [token for token, owner in zip(ids, owners) if owner == i]
            self.assertEqual(self.tokenizer.decode(pieces), (' ' if i else '') + word)

    def test_aggregation_and_first_word(self):
        class ConstantModel:
            config = SimpleNamespace(max_position_embeddings=1024)
            def __call__(self, input_ids, use_cache=False):
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 50257)))
        rows = [dict(item=1, zone=i+1, word=w) for i, w in enumerate(['extraordinarily', 'long-bearded', 'cat.'])]
        result = score_story(rows, self.tokenizer, ConstantModel(), 'cpu')
        self.assertTrue(math.isnan(result[0]['surprisal']))
        self.assertGreater(result[0]['n_bpe'], 1)
        for r in result[1:]:
            self.assertAlmostEqual(r['surprisal'], r['n_bpe'] * math.log2(50257), places=4)

    def test_full_corpus_alignment(self):
        regions = read_regions()
        count = 0
        for item in range(1, 11):
            words = [r['word'] for r in regions if r['item'] == item]
            ids, owners = encode_regions(words, self.tokenizer)
            count += sum(owners.count(i) > 1 for i in range(len(words)))
        print(f'Corpus: {len(regions)} regions; {count} multi-BPE words')


class AnalysisTests(unittest.TestCase):
    def test_complete_join_and_spillover(self):
        regions = read_regions()
        # Synthetic scores test the join only; never write them to results/.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tsv"
            write_tsv(path, [dict(r, surprisal=float("nan") if r["zone"] == 1 else r["zone"]) for r in reversed(regions)])
            rows, report = build_analysis(model_paths={"gpt2": path, "distilgpt2": path})
        self.assertEqual(report["participant_observations"], 848875)
        self.assertEqual(report["exclusions"], 0)
        self.assertEqual(len(rows), 10256)
        self.assertEqual(rows[0]["n_observations"], 84)
        self.assertAlmostEqual(rows[0]["mean_RT"], 578.9642857142857)
        for r in rows:
            if r["zone"] == 1:
                self.assertEqual(r["prev_word_length"], "")
            elif r["zone"] == 2:
                self.assertTrue(math.isnan(r["prev_gpt2_surprisal"]))
            else:
                self.assertEqual(r["prev_gpt2_surprisal"], r["zone"] - 1)


if __name__ == '__main__':
    unittest.main()
