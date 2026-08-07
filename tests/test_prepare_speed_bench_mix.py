import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_speed_bench_mix.py"
SPEC = importlib.util.spec_from_file_location("prepare_speed_bench_mix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class PrepareSpeedBenchMixTest(unittest.TestCase):
    def candidate(
        self,
        question_id: str,
        category: str,
        input_tokens: int,
    ) -> object:
        return PREPARE.Candidate(
            prompt=f"prompt-{question_id}",
            prompt_sha256=f"hash-{question_id}",
            question_id=question_id,
            subset="throughput_1k",
            category=category,
            sub_category="test",
            source="test-source",
            source_id=question_id,
            raw_prompt_tokens=input_tokens - 1,
            input_tokens=input_tokens,
        )

    def test_clean_prompt_filters_placeholder(self) -> None:
        self.assertIsNone(PREPARE.clean_prompt([PREPARE.PLACEHOLDER]))

    def test_clean_prompt_removes_repeated_padding(self) -> None:
        self.assertEqual(
            PREPARE.clean_prompt(
                ["A real question.\nAnswer now please.\nAnswer now please.\n"]
            ),
            "A real question.",
        )

    def test_selection_is_balanced_and_deterministic(self) -> None:
        candidates = [
            self.candidate(f"high-{index}", "high_entropy", index * 100)
            for index in range(1, 7)
        ] + [
            self.candidate(f"low-{index}", "low_entropy", index * 100 + 10)
            for index in range(1, 7)
        ]

        first = PREPARE.select_balanced(candidates, 4)
        second = PREPARE.select_balanced(list(reversed(candidates)), 4)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(
            {category: sum(item.category == category for item in first)
             for category in ("high_entropy", "low_entropy")},
            {"high_entropy": 2, "low_entropy": 2},
        )
        self.assertEqual(len({item.input_tokens for item in first}), 4)

    def test_all_eligible_selection_is_complete_and_deterministic(self) -> None:
        candidates = [
            self.candidate("third", "low_entropy", 300),
            self.candidate("first", "high_entropy", 100),
            self.candidate("second", "low_entropy", 200),
        ]

        first = PREPARE.select_candidates(candidates, None)
        second = PREPARE.select_candidates(list(reversed(candidates)), None)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(candidates))
        self.assertEqual([item.input_tokens for item in first], [100, 200, 300])

    def test_random_selection_is_stable_and_not_length_quantiles(self) -> None:
        candidates = [
            self.candidate(str(index), "category", index * 100)
            for index in range(1, 11)
        ]

        first = PREPARE.select_random(candidates, 4, 42)
        second = PREPARE.select_random(list(reversed(candidates)), 4, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({item.prompt_sha256 for item in first}), 4)


if __name__ == "__main__":
    unittest.main()
