"""
Tests for the PA Diagnosis module.

Covers the three key modifications integrated into the combined guidelines:
  1. AVS is optional (waived under defined conditions).
  2. CXCR4 imaging is recommended when lateralization is ambiguous.
  3. Confirmation tests may be skipped for unambiguous clinical presentations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pa_diagnosis.assistant import PADiagnosisAssistant, PatientInfo
from pa_diagnosis.guidelines import COMBINED_PA_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helper to build a typical ARR-positive patient
# ---------------------------------------------------------------------------

def _arr_positive_patient(**kwargs):
    defaults = dict(
        age=50,
        systolic_bp=165,
        diastolic_bp=105,
        on_antihypertensives=True,
        antihypertensive_count=3,
        diuretic_included=True,
        serum_potassium=3.0,
        plasma_aldosterone=22.0,
        plasma_renin_activity=0.5,
        arr=44.0,
        willing_for_surgery=True,
    )
    defaults.update(kwargs)
    return PatientInfo(**defaults)


assistant = PADiagnosisAssistant()


# ===========================================================================
# System prompt tests
# ===========================================================================

def test_system_prompt_is_non_empty():
    assert COMBINED_PA_SYSTEM_PROMPT.strip(), "System prompt must not be empty"


def test_system_prompt_contains_avs_modification():
    """The combined prompt must document the AVS-optional rule."""
    assert (
        "AVS实验可以不做" in COMBINED_PA_SYSTEM_PROMPT
        or "AVS豁免" in COMBINED_PA_SYSTEM_PROMPT
        or "AVS是可选" in COMBINED_PA_SYSTEM_PROMPT
    ), "Combined prompt must describe the AVS-optional modification"


def test_system_prompt_contains_cxcr4_modification():
    """The combined prompt must mention CXCR4 imaging."""
    assert "CXCR4" in COMBINED_PA_SYSTEM_PROMPT, \
        "Combined prompt must reference CXCR4 imaging for ambiguous lateralization"


def test_system_prompt_contains_skip_confirmation_modification():
    """The combined prompt must explain that confirmation tests can be skipped."""
    assert "可以不进行确诊" in COMBINED_PA_SYSTEM_PROMPT or \
           "可跳过确诊试验" in COMBINED_PA_SYSTEM_PROMPT or \
           "豁免确诊" in COMBINED_PA_SYSTEM_PROMPT, \
        "Combined prompt must describe the skip-confirmation-test modification"


def test_system_prompt_contains_both_guideline_references():
    """The prompt must cite both the 2016 CMA and 2025 ES guidelines."""
    assert "2016" in COMBINED_PA_SYSTEM_PROMPT, "Must reference the 2016 CMA guideline"
    assert "2025" in COMBINED_PA_SYSTEM_PROMPT, "Must reference the 2025 ES guideline"


# ===========================================================================
# Screening indication tests
# ===========================================================================

def test_screening_indicated_for_hypertensive_patient():
    p = PatientInfo(systolic_bp=160, diastolic_bp=100)
    result = assistant.assess_screening_indication(p)
    assert result["indicated"] is True


def test_screening_indicated_for_spontaneous_hypokalemia():
    p = PatientInfo(spontaneous_hypokalemia=True, serum_potassium=2.8)
    result = assistant.assess_screening_indication(p)
    assert result["indicated"] is True


def test_screening_not_indicated_without_data():
    p = PatientInfo()
    result = assistant.assess_screening_indication(p)
    assert result["indicated"] is False


# ===========================================================================
# ARR evaluation tests
# ===========================================================================

def test_arr_positive_pra_units():
    p = PatientInfo(arr=35.0, renin_unit="PRA")
    assert assistant.check_arr_positive(p) is True


def test_arr_negative_pra_units():
    p = PatientInfo(arr=25.0, renin_unit="PRA")
    assert assistant.check_arr_positive(p) is False


def test_arr_positive_drc_units():
    p = PatientInfo(arr=4.0, renin_unit="DRC")
    assert assistant.check_arr_positive(p) is True


def test_arr_negative_drc_units():
    p = PatientInfo(arr=3.0, renin_unit="DRC")
    assert assistant.check_arr_positive(p) is False


def test_arr_returns_none_when_missing():
    p = PatientInfo()
    assert assistant.check_arr_positive(p) is None


# ===========================================================================
# Modification 3 – Skip confirmation test
# ===========================================================================

def test_skip_confirmation_unambiguous_cma_criteria():
    """
    Modification 3: spontaneous hypokalemia + renin below detection limit
    + aldosterone > 20 ng/dl should allow skipping confirmation tests.
    """
    p = PatientInfo(
        spontaneous_hypokalemia=True,
        renin_below_detection_limit=True,
        plasma_aldosterone=25.0,
    )
    result = assistant.can_skip_confirmation_test(p)
    assert result["can_skip"] is True, \
        "Should skip confirmation test when CMA direct-diagnosis criteria are met"
    assert result["reason"], "Should provide a reason for skipping"


def test_skip_confirmation_patient_unwilling_for_surgery():
    """
    Modification 3: patients unwilling/unable to have surgery bypass the full
    workup and go directly to MRA treatment.
    """
    p = PatientInfo(willing_for_surgery=False, arr=40.0)
    result = assistant.can_skip_confirmation_test(p)
    assert result["can_skip"] is True


def test_no_skip_confirmation_for_standard_patient():
    """Standard ARR-positive patient without unambiguous criteria must proceed
    through a confirmation test."""
    p = _arr_positive_patient(
        spontaneous_hypokalemia=False,
        renin_below_detection_limit=False,
        plasma_aldosterone=15.0,
        willing_for_surgery=True,
    )
    result = assistant.can_skip_confirmation_test(p)
    assert result["can_skip"] is False


# ===========================================================================
# Modification 1 – AVS optional / waived
# ===========================================================================

def test_avs_waived_green_channel():
    """
    Modification 1: age < 35 + severe PA + unilateral adenoma -> AVS waived.
    """
    p = PatientInfo(
        age=28,
        spontaneous_hypokalemia=True,
        renin_below_detection_limit=True,
        plasma_aldosterone=30.0,
        ct_adrenal="左侧单侧腺瘤 1.5cm",
        willing_for_surgery=True,
    )
    result = assistant.assess_avs_necessity(p)
    assert result["avs_waived"] is True, \
        "AVS should be waived under green-channel criteria (age<35, severe PA, unilateral adenoma)"
    assert result["avs_required"] is False


def test_avs_waived_unwilling_surgery():
    """
    Modification 1: patient not willing for surgery -> AVS is not needed.
    """
    p = PatientInfo(willing_for_surgery=False)
    result = assistant.assess_avs_necessity(p)
    assert result["avs_waived"] is True
    assert result["avs_required"] is False


def test_avs_required_typical_surgical_candidate():
    """Standard surgical candidate without green-channel criteria still needs AVS."""
    p = _arr_positive_patient(
        age=55,
        ct_adrenal="双侧轻度增生",
        spontaneous_hypokalemia=False,
        renin_below_detection_limit=False,
    )
    result = assistant.assess_avs_necessity(p)
    assert result["avs_required"] is True
    assert result["avs_waived"] is False


# ===========================================================================
# Modification 2 – CXCR4 imaging for ambiguous lateralization
# ===========================================================================

def test_cxcr4_recommended_when_avs_failed():
    """
    Modification 2: when AVS fails, CXCR4 imaging should be recommended.
    """
    p = _arr_positive_patient(avs_result="操作失败", ct_adrenal="双侧结节")
    result = assistant.assess_avs_necessity(p)
    assert result["recommend_cxcr4"] is True, \
        "CXCR4 imaging should be recommended when AVS has failed"
    assert "CXCR4" in result["cxcr4_reason"]


def test_cxcr4_recommended_when_ct_bilateral():
    """
    Modification 2: bilateral CT findings with ambiguous lateralization
    should trigger CXCR4 recommendation.
    """
    p = _arr_positive_patient(ct_adrenal="双侧肾上腺结节不确定优势侧")
    result = assistant.assess_avs_necessity(p)
    assert result["recommend_cxcr4"] is True


def test_cxcr4_not_recommended_when_lateralization_clear():
    """No CXCR4 recommendation when lateralization is not ambiguous."""
    p = _arr_positive_patient(
        age=55,
        ct_adrenal="左侧单侧腺瘤 2.0cm",
        avs_result=None,
        spontaneous_hypokalemia=False,
        renin_below_detection_limit=False,
    )
    result = assistant.assess_avs_necessity(p)
    assert result["recommend_cxcr4"] is False


# ===========================================================================
# Summary generation tests
# ===========================================================================

def test_generate_summary_returns_string():
    p = _arr_positive_patient()
    summary = assistant.generate_summary(p)
    assert isinstance(summary, str) and summary.strip()


def test_generate_summary_contains_all_sections():
    p = _arr_positive_patient()
    summary = assistant.generate_summary(p)
    assert "阶段研判" in summary
    assert "干扰物" in summary
    assert "诊断路径" in summary
    assert "干预" in summary or "滴定" in summary
    assert "红线" in summary or "告警" in summary


def test_generate_summary_skips_confirmation_when_indicated():
    p = PatientInfo(
        spontaneous_hypokalemia=True,
        renin_below_detection_limit=True,
        plasma_aldosterone=25.0,
        arr=50.0,
        renin_unit="PRA",
        willing_for_surgery=True,
    )
    summary = assistant.generate_summary(p)
    assert "跳过确诊试验" in summary or "可跳过" in summary


def test_generate_summary_recommends_cxcr4_when_needed():
    p = _arr_positive_patient(avs_result="操作失败", ct_adrenal="双侧结节")
    summary = assistant.generate_summary(p)
    assert "CXCR4" in summary


if __name__ == "__main__":
    import traceback

    tests = [
        test_system_prompt_is_non_empty,
        test_system_prompt_contains_avs_modification,
        test_system_prompt_contains_cxcr4_modification,
        test_system_prompt_contains_skip_confirmation_modification,
        test_system_prompt_contains_both_guideline_references,
        test_screening_indicated_for_hypertensive_patient,
        test_screening_indicated_for_spontaneous_hypokalemia,
        test_screening_not_indicated_without_data,
        test_arr_positive_pra_units,
        test_arr_negative_pra_units,
        test_arr_positive_drc_units,
        test_arr_negative_drc_units,
        test_arr_returns_none_when_missing,
        test_skip_confirmation_unambiguous_cma_criteria,
        test_skip_confirmation_patient_unwilling_for_surgery,
        test_no_skip_confirmation_for_standard_patient,
        test_avs_waived_green_channel,
        test_avs_waived_unwilling_surgery,
        test_avs_required_typical_surgical_candidate,
        test_cxcr4_recommended_when_avs_failed,
        test_cxcr4_recommended_when_ct_bilateral,
        test_cxcr4_not_recommended_when_lateralization_clear,
        test_generate_summary_returns_string,
        test_generate_summary_contains_all_sections,
        test_generate_summary_skips_confirmation_when_indicated,
        test_generate_summary_recommends_cxcr4_when_needed,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
