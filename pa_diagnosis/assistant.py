"""
PA Diagnosis Assistant
======================
A clinical decision-support assistant for Primary Aldosteronism (PA) that
combines the 2016 Chinese Medical Association guidelines and the 2025
Endocrine Society Clinical Practice Guidelines.

Key modifications vs. the original single-guideline prompts:
1. AVS is optional – waived for patients meeting defined criteria.
2. CXCR4-targeted imaging ([68Ga]Ga-Pentixafor PET/CT) is offered when
   lateralization remains ambiguous after conventional workup.
3. Confirmation (suppression) tests may be skipped for patients whose
   presentation already meets the unambiguous clinical criteria for PA.
"""

from dataclasses import dataclass, field
from typing import Optional

from .guidelines import COMBINED_PA_SYSTEM_PROMPT


@dataclass
class PatientInfo:
    """Structured container for patient clinical data."""

    # Demographics
    age: Optional[int] = None

    # Blood pressure (mmHg)
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    on_antihypertensives: bool = False
    antihypertensive_count: Optional[int] = None  # number of current agents
    diuretic_included: bool = False

    # Laboratory values
    serum_potassium: Optional[float] = None         # mmol/L
    plasma_aldosterone: Optional[float] = None      # ng/dl
    plasma_renin_activity: Optional[float] = None   # ng·ml⁻¹·h⁻¹ (PRA) or mU/L (DRC)
    renin_unit: str = "PRA"                         # "PRA" or "DRC"
    arr: Optional[float] = None

    # Imaging
    ct_adrenal: Optional[str] = None    # e.g. "left unilateral adenoma 1.2cm"
    avs_result: Optional[str] = None   # e.g. "failed", "bilateral", "left dominant"

    # Clinical flags
    spontaneous_hypokalemia: bool = False
    renin_below_detection_limit: bool = False
    family_history_pa: bool = False
    family_history_early_stroke: bool = False
    osa: bool = False

    # Patient preference
    willing_for_surgery: bool = False

    # Free-text additional information
    current_medications: list = field(default_factory=list)
    notes: str = ""


class PADiagnosisAssistant:
    """
    Clinical AI assistant for Primary Aldosteronism diagnosis that follows
    the combined 2016 CMA + 2025 ES guidelines with three key modifications:

    Modification 1 – AVS is optional:
        AVS may be waived when any of the following apply:
        * Patient age < 35 with severe PA features and CT showing a unilateral
          adenoma > 1.0 cm (2025 ES green channel).
        * Patient unwilling or unable to undergo surgery.
        * Lateralization is already evident and risk of AVS outweighs benefit.

    Modification 2 – CXCR4 imaging for ambiguous lateralization:
        When AVS is technically unsuccessful, results are discordant, or CT/AVS
        cannot confidently determine the dominant side, recommend
        [68Ga]Ga-Pentixafor PET/CT (CXCR4-targeted imaging).

    Modification 3 – Confirmation tests may be skipped:
        Patients with unambiguous clinical indications (spontaneous hypokalemia +
        renin below detection limit + aldosterone > 20 ng/dl, or high-probability
        unilateral PA per 2025 ES guidelines) may proceed directly to subtype
        workup without a suppression test.
    """

    def __init__(self):
        self.system_prompt = COMBINED_PA_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return the combined PA guidelines system prompt."""
        return self.system_prompt

    def assess_screening_indication(self, patient: PatientInfo) -> dict:
        """
        Step 1 – Determine whether PA screening is indicated.

        Returns a dict with keys:
          indicated (bool), reasons (list[str])
        """
        reasons = []

        # 2025 ES: universal screening for all hypertensive patients
        if patient.systolic_bp is not None or patient.diastolic_bp is not None:
            reasons.append(
                "2025 ES指南：所有确诊高血压患者均应进行PA筛查（普遍筛查原则）"
            )

        # 2016 CMA specific indications
        if (
            patient.systolic_bp is not None
            and patient.diastolic_bp is not None
            and patient.systolic_bp > 150
            and patient.diastolic_bp > 100
        ):
            reasons.append("持续性高血压（>150/100 mmHg）")

        if (
            patient.on_antihypertensives
            and patient.antihypertensive_count is not None
            and patient.antihypertensive_count >= 3
            and patient.diuretic_included
        ):
            reasons.append("难治性高血压")

        if patient.spontaneous_hypokalemia:
            reasons.append("高血压合并自发性低钾血症")

        if patient.family_history_pa:
            reasons.append("原醛症患者的一级亲属且合并高血压")

        if patient.family_history_early_stroke:
            reasons.append("早发（<40岁）脑血管意外家族史的高血压患者")

        if patient.osa:
            reasons.append("高血压合并阻塞性睡眠呼吸暂停（OSA）")

        return {"indicated": bool(reasons), "reasons": reasons}

    def check_arr_positive(self, patient: PatientInfo) -> Optional[bool]:
        """
        Step 2 – Evaluate whether ARR is positive.

        Returns True/False, or None if data are insufficient.
        """
        if patient.arr is None:
            return None

        cutoff = 30.0 if patient.renin_unit == "PRA" else 3.7
        return patient.arr >= cutoff

    def _has_severe_pa_features(self, patient: PatientInfo) -> bool:
        """
        Return True when the patient shows unambiguous severe-PA features shared
        by both the 2016 CMA direct-diagnosis rule and the 2025 ES high-probability
        unilateral PA criterion:
          * spontaneous hypokalemia
          * plasma renin below the detection limit
          * plasma aldosterone > 20 ng/dl
        """
        return (
            patient.spontaneous_hypokalemia
            and patient.renin_below_detection_limit
            and patient.plasma_aldosterone is not None
            and patient.plasma_aldosterone > 20
        )

    def can_skip_confirmation_test(self, patient: PatientInfo) -> dict:
        """
        Step 3 (Modification 3) – Determine whether confirmation tests may be
        skipped based on unambiguous clinical criteria.

        Returns a dict with keys:
          can_skip (bool), reason (str)
        """
        # Criterion 1 (shared by 2016 CMA and 2025 ES): unambiguous severe-PA
        # features – spontaneous hypokalemia + renin below detection limit +
        # aldosterone > 20 ng/dl.
        if self._has_severe_pa_features(patient):
            return {
                "can_skip": True,
                "reason": (
                    "直接确诊条件满足（2016中华医学会共识 及 2025 ES指南高概率单侧PA标准）："
                    "自发性低钾血症 + 血浆肾素低于检测下限 + 醛固酮 >20 ng/dl，"
                    "可跳过确诊试验直接确诊PA，进入CT+AVS分型。"
                ),
            }

        # Criterion 2: patient unwilling/unable to have surgery – go straight to MRA
        if not patient.willing_for_surgery:
            return {
                "can_skip": True,
                "reason": (
                    "患者无手术意愿或手术不可行，直接进入MRA药物治疗阶段，"
                    "无需完成确诊试验及分型检查（2025 ES指南以患者意愿为中心原则）。"
                ),
            }

        return {"can_skip": False, "reason": ""}

    def assess_avs_necessity(self, patient: PatientInfo) -> dict:
        """
        Step 4 (Modification 1) – Determine whether AVS is necessary or may
        be waived.

        Returns a dict with keys:
          avs_required (bool), avs_waived (bool), reason (str),
          recommend_cxcr4 (bool), cxcr4_reason (str)
        """
        # Not relevant if patient does not want surgery
        if not patient.willing_for_surgery:
            return {
                "avs_required": False,
                "avs_waived": True,
                "reason": "患者无手术意愿，无需AVS。",
                "recommend_cxcr4": False,
                "cxcr4_reason": "",
            }

        # Green-channel exemption: age < 35, severe PA, unilateral adenoma > 1.0 cm
        ct_has_unilateral_adenoma_over_1cm = (
            patient.ct_adrenal is not None
            and "单侧" in patient.ct_adrenal
            and "腺瘤" in patient.ct_adrenal
        )
        severe_pa = self._has_severe_pa_features(patient)

        if (
            patient.age is not None
            and patient.age < 35
            and severe_pa
            and ct_has_unilateral_adenoma_over_1cm
        ):
            return {
                "avs_required": False,
                "avs_waived": True,
                "reason": (
                    "绿色通道豁免AVS：年龄<35岁 + 重度PA特征 + CT显示单侧腺瘤。"
                    "可直接行腹腔镜单侧肾上腺切除术（2025 ES指南及2016中华医学会共识）。"
                ),
                "recommend_cxcr4": False,
                "cxcr4_reason": "",
            }

        # Modification 2: recommend CXCR4 imaging when lateralization is ambiguous
        avs_ambiguous = patient.avs_result is not None and patient.avs_result in (
            "failed",
            "ambiguous",
            "bilateral unclear",
            "操作失败",
            "双侧不清",
            "结果不一致",
        )
        ct_bilateral_or_uncertain = patient.ct_adrenal is not None and (
            "双侧" in patient.ct_adrenal or "不确定" in patient.ct_adrenal
        )

        recommend_cxcr4 = avs_ambiguous or ct_bilateral_or_uncertain
        cxcr4_reason = ""
        if recommend_cxcr4:
            cxcr4_reason = (
                "【Modification 2 – CXCR4显像】AVS操作失败/结果不一致，或CT显示双侧结节难以"
                "明确优势侧，推荐行 [⁶⁸Ga]Ga-Pentixafor PET/CT（CXCR4靶向显像）以获取"
                "功能性侧别定位依据，辅助手术决策。"
            )

        return {
            "avs_required": True,
            "avs_waived": False,
            "reason": (
                "患者具有手术意愿，CT提示形态异常或分型不明确，强推荐AVS以明确优势分泌侧。"
            ),
            "recommend_cxcr4": recommend_cxcr4,
            "cxcr4_reason": cxcr4_reason,
        }

    def generate_summary(self, patient: PatientInfo) -> str:
        """
        Generate a structured clinical decision summary for the given patient,
        following the five-section format mandated by the combined guidelines.
        """
        lines = []

        # ── Section 1: Stage assessment ───────────────────────────────
        lines.append("### 🩺 1.【阶段研判与指南定位】")
        screening = self.assess_screening_indication(patient)
        if screening["indicated"]:
            lines.append("当前阶段：**PA筛查期**")
            for r in screening["reasons"]:
                lines.append(f"  - {r}")
        else:
            lines.append("当前阶段：暂无明确PA筛查指征，请补充血压及相关临床资料。")

        # ── Section 2: Drug interference check ────────────────────────
        lines.append("\n### 💊 2.【干扰物雷达与基线清理】")
        if patient.current_medications:
            lines.append("当前用药：" + "、".join(patient.current_medications))
            lines.append(
                "请核查是否包含：螺内酯/MRAs（需停药4周）、ACEI/ARB/CCB（需停药2周）、"
                "β受体阻滞剂（假阳性风险，建议停药2周）。"
                "洗脱期替代药：肼屈嗪、非二氢吡啶类CCB、哌唑嗪。"
            )
        else:
            lines.append("未提供用药信息，请补充当前用药单以评估干扰因素。")

        # ── Section 3: Optimal diagnostic pathway ─────────────────────
        lines.append("\n### ⚖️ 3.【诊断路径最优解】")

        arr_pos = self.check_arr_positive(patient)
        if arr_pos is None:
            lines.append("ARR数据缺失，请先完成ARR筛查。")
        elif not arr_pos:
            lines.append(f"ARR={patient.arr}，低于切点，PA筛查阴性。建议定期随访。")
        else:
            lines.append(f"ARR={patient.arr}，初筛阳性，进入确诊评估。")

            skip_info = self.can_skip_confirmation_test(patient)
            if skip_info["can_skip"]:
                lines.append(f"✅ **可跳过确诊试验**：{skip_info['reason']}")
            else:
                lines.append(
                    "需进行确诊试验（生理盐水试验 / 卡托普利试验）。"
                    "注意：严重低钾、难治性高血压、心衰者禁用生理盐水试验。"
                )

            avs_info = self.assess_avs_necessity(patient)
            lines.append(f"\n**AVS评估**：{avs_info['reason']}")
            if avs_info["avs_waived"]:
                lines.append("  ✅ 已满足AVS豁免条件，可免除AVS检查。")
            if avs_info["recommend_cxcr4"]:
                lines.append(f"  🔬 {avs_info['cxcr4_reason']}")

        # ── Section 4: Intervention & titration ───────────────────────
        lines.append("\n### 🎯 4.【特异性干预与靶向滴定】")
        if not patient.willing_for_surgery:
            lines.append(
                "患者无手术意愿，启动MRA治疗：\n"
                "  - 首选**螺内酯**，起始20 mg/d，以血钾调量。\n"
                "  - 开始MRA治疗2-4天内停用口服补钾（除非重度低钾）。\n"
                "  - 强制低钠饮食（食盐<5g/天）。\n"
                "  - 每2-3个月复查电解质、肾功和肾素，以肾素是否解除抑制为核心滴定靶点。"
            )
        else:
            lines.append("待分型结果明确后，按醛固酮瘤/特醛症/GRA等亚型给予对应治疗。")

        # ── Section 5: Red-line alerts ─────────────────────────────────
        lines.append("\n### ⚠️ 5.【临床红线告警（Next Actions）】")
        alerts = []
        if patient.serum_potassium is not None and patient.serum_potassium < 3.5:
            alerts.append("低钾血症（K⁺<3.5 mmol/L）：尽量纠正血钾至正常范围后再行ARR检测。")
        if patient.ct_adrenal and ("腺瘤" in patient.ct_adrenal):
            alerts.append(
                "CT发现肾上腺腺瘤：强制追加1-mg过夜地塞米松抑制试验（1-mg DST）以排查皮质醇共分泌（ACS）。"
            )
        alerts.append("螺内酯副反应预警：乳腺发育（男性）、肾功能一过性下降（非肾衰指征）。")
        alerts.append("随访频率：MRA治疗开始后每2-3个月复查肾素、醛固酮、电解质及肾功能。")

        for alert in alerts:
            lines.append(f"  - {alert}")

        if patient.notes:
            lines.append(f"\n**补充说明**：{patient.notes}")

        return "\n".join(lines)
