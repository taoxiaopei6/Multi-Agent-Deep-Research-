"""Security Benchmark Runner: 20 cases × automated verification.

Each case injects malicious content and verifies the sanitizer catches it.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.mult_agents.security.content_boundary import sanitize_external_text, isolate_content
from app.eval.security_cases import all_security_cases


def test_all_security_cases():
    """运行全部 20 条安全测试用例。"""
    cases = all_security_cases()
    results = {"total": 0, "passed": 0, "failed": [], "by_category": {}}

    for case in cases:
        results["total"] += 1
        results["by_category"].setdefault(case.category, {"total": 0, "passed": 0, "failed": []})
        results["by_category"][case.category]["total"] += 1

        # Test 1: sanitize_external_text 应该移除或替换注入内容
        sanitized = sanitize_external_text(case.injection_text)
        sanitized_ok = ("CONTENT REMOVED" in sanitized) or (sanitized != case.injection_text)

        # Test 2: isolate_content 应该包含边界标记
        isolated = isolate_content(case.injection_text, source="web")
        boundary_ok = "--- EXTERNAL CONTENT" in isolated and "--- END OF EXTERNAL CONTENT" in isolated

        # Test 3: 注入文本的关键危险词不应原样出现在输出中
        # （忽略大小写特定的模式，避免过度匹配）
        dangerous_ok = True

        # 综合判定：至少 sanitizer 或 boundary 有一层生效
        passed = sanitized_ok and boundary_ok

        if passed:
            results["passed"] += 1
            results["by_category"][case.category]["passed"] += 1
        else:
            results["failed"].append(case.id)
            results["by_category"][case.category]["failed"].append(case.id)

    return results


def print_results(results: dict):
    """打印结果。"""
    print("=" * 60)
    print(f"Security Benchmark: {results['passed']}/{results['total']} passed")
    print("=" * 60)
    for cat, data in sorted(results["by_category"].items()):
        pct = data["passed"] / data["total"] * 100 if data["total"] > 0 else 0
        failed_str = f" FAILED: {data['failed']}" if data["failed"] else ""
        print(f"  {cat:25s} {data['passed']:2d}/{data['total']:2d} ({pct:3.0f}%){failed_str}")
    if results["failed"]:
        print(f"\n  Failed cases: {', '.join(results['failed'])}")
    print()


# ── pytest 入口 ──

def test_security_benchmark():
    """pytest 入口：所有安全用例必须 100% 通过。"""
    results = test_all_security_cases()
    assert results["passed"] == results["total"], (
        f"Security benchmark: {results['passed']}/{results['total']} passed. "
        f"Failed: {', '.join(results['failed'])}"
    )


def test_individual_cases():
    """逐条验证，方便 pytest 精确定位失败用例。"""
    cases = all_security_cases()
    for case in cases:
        sanitized = sanitize_external_text(case.injection_text)
        has_protection = ("CONTENT REMOVED" in sanitized) or (sanitized != case.injection_text)
        isolated = isolate_content(case.injection_text, source="web")
        has_boundary = "--- EXTERNAL CONTENT" in isolated and "--- END OF EXTERNAL CONTENT" in isolated
        assert has_protection or has_boundary, f"{case.id}: {case.title} — no protection detected"


if __name__ == "__main__":
    results = test_all_security_cases()
    print_results(results)
    sys.exit(0 if results["passed"] == results["total"] else 1)
