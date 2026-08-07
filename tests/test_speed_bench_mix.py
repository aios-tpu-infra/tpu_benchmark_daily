import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "speed_bench_mix"
MANIFEST = FIXTURES / "manifest.json"
FIXTURE_DATASET_SHA256 = (
    "5fca7ec7fc785d89f8d91c6653f0eb34297983370b530a4d1ded91c550093f76"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AGGREGATE = load_module(
    "aggregate_speed_bench_mix",
    PROJECT_ROOT / "scripts" / "aggregate_speed_bench_mix.py",
)
UPDATE_REPORT = load_module(
    "update_speed_bench_report",
    PROJECT_ROOT / "scripts" / "update_speed_bench_report.py",
)


class SpeedBenchMixTest(unittest.TestCase):
    def aggregate(self, **overrides):
        arguments = {
            "manifest_path": MANIFEST,
            "mode": "all",
            "concurrency_runs": [
                (8, "success", FIXTURES / "throughput.json")
            ],
        }
        arguments.update(overrides)
        return AGGREGATE.aggregate(**arguments)

    def test_aggregate_extracts_throughput_and_load_ttft(self) -> None:
        summary = self.aggregate()
        result = summary["throughput"]["results"][0]

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["benchmark"]["num_prompts"], 20)
        self.assertEqual(summary["benchmark"]["total_input_tokens"], 234598)
        self.assertAlmostEqual(
            result["input_token_throughput"],
            234598 / 13.568865343928337,
        )
        self.assertAlmostEqual(
            result["median_ttft_ms"],
            2955.2259200718254,
        )
        observations = result["observations"]
        self.assertEqual(len(observations), 20)
        self.assertEqual(observations[0]["input_tokens"], 1038)
        self.assertEqual(observations[-1]["input_tokens"], 32982)

    def test_load_manifest_validates_compressed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            content = b'{"prompt":"fixture"}\n'
            artifact = root / "requests.jsonl.gz"
            artifact.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {
                            "path": artifact.name,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "artifact_sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                            "requests": 1,
                            "total_input_tokens": 3,
                            "input_tokens": [3],
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest, loaded_artifact = AGGREGATE.load_manifest(manifest_path)

        self.assertEqual(loaded_artifact, artifact)
        self.assertEqual(
            manifest["dataset"]["sha256"], hashlib.sha256(content).hexdigest()
        )

    def test_mode_preserves_concurrency_sweep_results(self) -> None:
        summary = self.aggregate(mode="ttft")

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["throughput"]["status"], "success")
        self.assertEqual(summary["serial_ttft"]["status"], "not-run")

    def test_throughput_concurrency_is_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "throughput.json"
            data = json.loads(
                (FIXTURES / "throughput.json").read_text(encoding="utf-8")
            )
            data["max_concurrency"] = 64
            result_path.write_text(json.dumps(data), encoding="utf-8")
            summary = self.aggregate(
                concurrency_runs=[(64, "success", result_path)],
            )

        self.assertEqual(summary["status"], "success")
        self.assertEqual(
            summary["throughput"]["results"][0][
                "configured_max_concurrency"
            ],
            64,
        )

    def test_aggregate_keeps_each_requested_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "throughput_c64.json"
            data = json.loads(
                (FIXTURES / "throughput.json").read_text(encoding="utf-8")
            )
            data["max_concurrency"] = 64
            result_path.write_text(json.dumps(data), encoding="utf-8")
            summary = self.aggregate(
                concurrency_runs=[
                    (8, "success", FIXTURES / "throughput.json"),
                    (64, "success", result_path),
                ]
            )

        self.assertEqual(
            summary["throughput"]["requested_concurrencies"], [8, 64]
        )
        self.assertEqual(
            summary["throughput"]["configured_max_concurrency"], 64
        )
        self.assertEqual(
            [
                result["configured_max_concurrency"]
                for result in summary["throughput"]["results"]
            ],
            [8, 64],
        )
        self.assertTrue(
            all(
                result["p90_ttft_ms"] > 0
                for result in summary["throughput"]["results"]
            )
        )

    def test_aggregate_accepts_pcp8_and_preserves_dataset_hash(self) -> None:
        summary = self.aggregate(benchmark_config="pcp8")

        self.assertEqual(summary["benchmark"]["benchmark_config"], "pcp8")
        self.assertEqual(
            summary["benchmark"]["dataset_sha256"],
            FIXTURE_DATASET_SHA256,
        )

    def test_dataset_length_mismatch_is_recorded_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "throughput.json"
            data = json.loads(
                (FIXTURES / "throughput.json").read_text(encoding="utf-8")
            )
            data["input_lens"][0] = 999
            result_path.write_text(json.dumps(data), encoding="utf-8")
            summary = self.aggregate(
                concurrency_runs=[(8, "success", result_path)]
            )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["throughput"]["status"], "failed")
        self.assertIn(
            "dataset manifest",
            summary["throughput"]["results"][0]["error"],
        )

    def test_report_history_and_readme_are_updated_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "runs" / "20260806T060018Z"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "summary.json"
            summary_path.write_text(
                json.dumps(self.aggregate()), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-08-06T06:00:18+00:00",
                        "completed_at": "2026-08-06T06:02:34+00:00",
                        "machine_ip": "127.0.0.1",
                        "torchtpu_vllm_revision": "f53d6300e29f5d77",
                        "torch_tpu_version": "test",
                    }
                ),
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "# Test\n\n## Layout\n\nText.\n", encoding="utf-8"
            )
            record = UPDATE_REPORT.build_record(
                root, run_dir, summary_path, "Qwen3.5-397B-A17B-FP8"
            )
            runs = UPDATE_REPORT.update_history([], record)
            pcp_summary_path = run_dir / "pcp-summary.json"
            pcp_summary_path.write_text(
                json.dumps(self.aggregate(benchmark_config="pcp8")),
                encoding="utf-8",
            )
            pcp_record = UPDATE_REPORT.build_record(
                root,
                run_dir,
                pcp_summary_path,
                "Qwen3.5-397B-A17B-FP8",
            )
            runs = UPDATE_REPORT.update_history(runs, pcp_record)
            (root / "reports").mkdir()
            UPDATE_REPORT.atomic_write(
                root / "reports" / "speed_bench_history.json",
                UPDATE_REPORT.render_history(runs),
            )
            UPDATE_REPORT.update_readme(
                root / "README.md", UPDATE_REPORT.render_readme_block(runs, 10)
            )

            history = json.loads(
                (root / "reports" / "speed_bench_history.json").read_text()
            )
            readme = (root / "README.md").read_text()

        self.assertEqual(len(history["runs"]), 2)
        self.assertEqual(
            {item["completed_at"] for item in history["runs"]},
            {"2026-08-06T06:02:34+00:00"},
        )
        self.assertEqual(
            {item["benchmark_config"] for item in history["runs"]},
            {"dp8", "pcp8"},
        )
        self.assertIn("SPEED_BENCH_REPORT_START", readme)
        self.assertIn("Real variable-length prefill benchmark", readme)
        self.assertIn("| Prefill mode | Dataset SHA-256 |", readme)
        self.assertIn("TTFT P50 (ms) | TTFT P90 (ms) | TTFT P99 (ms)", readme)
        self.assertIn(f"**DP8** | `{FIXTURE_DATASET_SHA256[:12]}`", readme)
        self.assertIn(f"**PCP8** | `{FIXTURE_DATASET_SHA256[:12]}`", readme)
        self.assertLess(
            readme.index("SPEED_BENCH_REPORT_START"), readme.index("## Layout")
        )
        csv_report = UPDATE_REPORT.render_csv(runs)
        self.assertTrue(csv_report.startswith("run_id,benchmark_config,"))
        self.assertNotIn("\r", csv_report)

    def test_report_history_keeps_dp8_and_pcp8_from_the_same_run(self) -> None:
        common = {
            "run_id": "20260806T070000Z",
            "completed_at": "2026-08-06T07:10:00+00:00",
            "dataset_sha256": FIXTURE_DATASET_SHA256,
        }
        legacy_dp = {
            **common,
            "summary_path": "runs/shared/results/dp8/speed_bench_mix/summary.json",
        }
        pcp = {
            **common,
            "benchmark_config": "pcp8",
            "summary_path": "runs/shared/results/pcp8/speed_bench_mix/summary.json",
        }

        runs = UPDATE_REPORT.update_history([legacy_dp], pcp)

        self.assertEqual(len(runs), 2)
        self.assertTrue(all("benchmark_config" in item for item in runs))
        self.assertEqual(
            {
                UPDATE_REPORT.normalize_benchmark_config(
                    item.get("benchmark_config"), legacy_default=True
                )
                for item in runs
            },
            {"dp8", "pcp8"},
        )


if __name__ == "__main__":
    unittest.main()
