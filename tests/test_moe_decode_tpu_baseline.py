"""Opt-in full-size DP4/TP2/EP8 MoE parity checks on eight TPU chiplets.

Run with RUN_MOE_TPU_UT=1; intentionally opt-in because the test owns all TPUs
and allocates a complete MoE layer. Stop the serving process before running.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_MOE_TPU_UT') != '1',
    reason='set RUN_MOE_TPU_UT=1 to run the eight-chiplet hardware UT',
)


@pytest.fixture(scope='module')
def baseline():
    spec = importlib.util.spec_from_file_location(
        'bench_moe_decode_ut', ROOT / 'scripts/bench_moe_decode_ut.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Baseline(seed=20260921)


@pytest.mark.parametrize('pool', [16, 32, 64])
def test_decode_moe_matches_dense_reference(baseline, pool, tmp_path):
    """Validate real distributed computation, including partial expert hits.

    The oracle uses independent dense contractions and token-order weighted
    accumulation, without production routing sort/permute/unpermute kernels.
    """
    baseline.set_pool(pool)
    result = baseline.check(tmp_path / f'pool{pool}')
    assert result['tp_replicas_bitwise_equal']
    assert result['route_count'] == 256 * 10
    assert all(0 < count <= pool for count in result['active_experts_per_rank'])
