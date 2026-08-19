"""Regression tests for SM120 (RTX Pro 6000) kernel-selection guards.

Several SGLang fast paths are implemented with instructions that exist only on
*datacenter* Blackwell (SM100, compute-capability major 10): ``tcgen05``, TMEM
and ``cp.async.bulk``. Consumer/datacenter-class SM120 reports a *higher*
capability major (12) but lacks those instructions. Guards written as
``major >= 10`` therefore wrongly enabled the SM100-only kernels on SM120,
where ``ptxas`` fails to JIT-compile them (e.g. ``Instruction 'tcgen05.alloc'
not supported on .target 'sm_120a'``), crashing multi-node serving.

These tests pin the contract that the tcgen05/TMA-gated paths select on
*SM100 exactly* and fall back to the portable (Triton) pipeline on every other
architecture (SM90, SM120, ...). They run on CPU by mocking the device
capability, so they need no GPU.
"""

import inspect
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (major, minor) for the architectures we care about, mirroring
# torch.cuda.get_device_capability().
_SM90 = (9, 0)  # Hopper (H100)
_SM100 = (10, 0)  # datacenter Blackwell (B100/B200/GB200)
_SM120 = (12, 0)  # RTX Pro 6000 (GCP G4) -- no tcgen05/TMA

HIDDEN_SIZE_K3 = 7168  # Kimi-K3 attention-residual hidden size


def _reset_attn_residual_cache():
    # _use_fast caches its capability probe in a module global; reset between
    # capability mocks.
    import sglang.srt.layers.attn_residual as ar

    ar._FAST_SUPPORTED = None


class TestAttnResidualUseFast(unittest.TestCase):
    """attn_residual._use_fast must be SM100-only, not SM100+."""

    def _use_fast(self, capability):
        import sglang.srt.layers.attn_residual as ar

        _reset_attn_residual_cache()
        # is_sm100_supported() reads torch.cuda.get_device_capability(); patch
        # the helper at its use site so the mock takes effect regardless of the
        # lru_cache on the helper itself.
        with patch.object(
            ar, "is_sm100_supported", return_value=(capability[0] == 10)
        ):
            return ar._use_fast(HIDDEN_SIZE_K3)

    def test_sm100_uses_fast_tma_kernel(self):
        self.assertTrue(self._use_fast(_SM100))

    def test_sm120_falls_back_to_triton(self):
        # The reported bug: SM120 (major 12) satisfied `major >= 10` and tried
        # to JIT the tcgen05 TMA kernel, failing ptxas. Must now fall back.
        self.assertFalse(self._use_fast(_SM120))

    def test_sm90_falls_back_to_triton(self):
        self.assertFalse(self._use_fast(_SM90))

    def test_source_does_not_use_major_ge_10(self):
        import sglang.srt.layers.attn_residual as ar

        src = inspect.getsource(ar._use_fast)
        self.assertNotIn(">= 10", src)
        self.assertIn("is_sm100_supported", src)


class TestLinearAttnCutedslPrefillGate(unittest.TestCase):
    """The CuTe DSL GDN/KDA prefill kernels are tcgen05/TMA-based (SM100-only)."""

    def _gate(self, module_path, capability):
        import importlib

        mod = importlib.import_module(module_path)
        with patch.object(
            mod, "is_sm100_supported", return_value=(capability[0] == 10)
        ):
            return mod._is_blackwell()

    def test_gdn_cutedsl_sm100_supported(self):
        self.assertTrue(
            self._gate(
                "sglang.srt.layers.attention.linear.kernels.gdn_cutedsl", _SM100
            )
        )

    def test_gdn_cutedsl_sm120_falls_back(self):
        self.assertFalse(
            self._gate(
                "sglang.srt.layers.attention.linear.kernels.gdn_cutedsl", _SM120
            )
        )

    def test_kda_cutedsl_sm100_supported(self):
        self.assertTrue(
            self._gate(
                "sglang.srt.layers.attention.linear.kernels.kda_cutedsl", _SM100
            )
        )

    def test_kda_cutedsl_sm120_falls_back(self):
        self.assertFalse(
            self._gate(
                "sglang.srt.layers.attention.linear.kernels.kda_cutedsl", _SM120
            )
        )

    def test_cutedsl_sources_do_not_use_major_ge_10(self):
        import importlib

        for path in (
            "sglang.srt.layers.attention.linear.kernels.gdn_cutedsl",
            "sglang.srt.layers.attention.linear.kernels.kda_cutedsl",
        ):
            mod = importlib.import_module(path)
            src = inspect.getsource(mod._is_blackwell)
            self.assertNotIn(">= 10", src, msg=f"{path} still uses major >= 10")
            self.assertIn("is_sm100_supported", src)


if __name__ == "__main__":
    unittest.main()
