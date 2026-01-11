"""
PRC – Timeline Verification Module (CORRIGÉ)
---------------------------------
Corrections:
1. Requête SQL dynamic_events (dict pas array)
2. Analyse fallback vs vraies données
3. Détection proxy linéaire
"""

import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any

# ================================
# CONFIGURATION
# ================================

DB_PATH = Path("./prc_automation/prc_database/prc_r0_results.db")
GAMMA_PROFILES_CSV = Path("./reports/verdicts/2/gamma_profiles.csv")
GAMMA_PROFILES_JSON = Path("./reports/verdicts/2/gamma_profiles.json")
DIAGNOSTICS_JSON = Path("./reports/verdicts/2/diagnostics.json")

OUTPUT_JSON = Path("timeline_verification.json")

# Seuils R0 (volontairement permissifs)
MIN_EVENT_PRESENCE_RATIO = 0.01   # 1% des observations avec événements suffit
MAX_NO_TIMELINE_RATIO_OK = 0.95   # Jusqu'à 95% de no_significant_dynamics est acceptable

# ================================
# UTILITAIRES
# ================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================
# CHECKS
# ================================

def check_dynamic_events(db_path: Path) -> Dict[str, Any]:
    """
    Vérifie la présence minimale de dynamic_events dans la DB.
    
    FIX: dynamic_events est un DICT, pas un array.
    Structure: {"metric_name": {"deviation_onset": ..., "sequence": [...]}}
    """
    if not db_path.exists():
        return {
            "status": "FAIL",
            "reason": "Database not found",
            "event_ratio": 0.0,
        }

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM TestObservations")
    total = cur.fetchone()[0]

    # FIX: Vérifier que dynamic_events contient au moins 1 clé (métrique)
    cur.execute(
        """
        SELECT COUNT(*) FROM TestObservations
        WHERE json_type(json_extract(observation_data, '$.dynamic_events')) = 'object'
          AND json_extract(observation_data, '$.dynamic_events') != '{}'
        """
    )
    with_events = cur.fetchone()[0]
    
    # NOUVEAU: Vérifier présence timeseries (vraies données vs fallback)
    cur.execute(
        """
        SELECT COUNT(*) FROM TestObservations
        WHERE json_type(json_extract(observation_data, '$.timeseries')) = 'object'
          AND json_extract(observation_data, '$.timeseries') != '{}'
        """
    )
    with_timeseries = cur.fetchone()[0]
    
    conn.close()

    ratio = with_events / max(total, 1)
    timeseries_ratio = with_timeseries / max(total, 1)

    if ratio == 0:
        status = "FAIL"
        reason = "No dynamic_events detected in any observation"
    elif ratio < MIN_EVENT_PRESENCE_RATIO:
        status = "WARN"
        reason = "Dynamic events detected but extremely sparse"
    else:
        status = "PASS"
        reason = "Dynamic events present"

    return {
        "status": status,
        "reason": reason,
        "total_observations": total,
        "observations_with_events": with_events,
        "event_ratio": ratio,
        "observations_with_timeseries": with_timeseries,
        "timeseries_ratio": timeseries_ratio,
        "fallback_likely": timeseries_ratio < 0.1 and ratio == 0  # Fallback si pas de vraies données
    }


def check_timelines(gamma_profiles_json: Path) -> Dict[str, Any]:
    """Analyse la distribution des timelines générées."""
    profiles = load_json(gamma_profiles_json)

    timeline_counter = Counter()

    for gamma_data in profiles.values():
        tests = gamma_data.get("tests", {})
        for test_data in tests.values():
            timeline = test_data.get("timeline", "UNKNOWN")
            timeline_counter[timeline] += 1

    total = sum(timeline_counter.values())
    no_dyn = timeline_counter.get("no_significant_dynamics", 0)
    ratio_no_dyn = no_dyn / max(total, 1)

    # Détection de timelines compositionnelles
    compositionals = [
        tl for tl in timeline_counter
        if any(k in tl for k in ("early_", "mid_", "late_"))
    ]

    if total == 0:
        status = "FAIL"
        reason = "No timelines found"
    elif ratio_no_dyn > MAX_NO_TIMELINE_RATIO_OK and not compositionals:
        status = "WARN"
        reason = "Timelines exist but no compositionnal patterns detected"
    else:
        status = "PASS"
        reason = "Timelines generated correctly"

    return {
        "status": status,
        "reason": reason,
        "total_timelines": total,
        "no_significant_dynamics_ratio": ratio_no_dyn,
        "timeline_distribution": dict(timeline_counter),
        "compositionnal_examples": compositionals[:10],
    }


def check_trivial_regimes(gamma_profiles_json: Path) -> Dict[str, Any]:
    """Quantifie la présence des régimes TRIVIAL sans les juger."""
    profiles = load_json(gamma_profiles_json)

    regime_counter = Counter()

    for gamma_data in profiles.values():
        for test_data in gamma_data.get("tests", {}).values():
            regime = test_data.get("regime", "UNKNOWN")
            regime_counter[regime] += 1

    total = sum(regime_counter.values())
    trivial_ratio = regime_counter.get("TRIVIAL", 0) / max(total, 1)

    status = "PASS"
    reason = "Regime distribution collected"

    return {
        "status": status,
        "reason": reason,
        "total_profiles": total,
        "regime_distribution": dict(regime_counter),
        "trivial_ratio": trivial_ratio,
    }


def analyze_fallback_usage(gamma_profiles_json: Path) -> Dict[str, Any]:
    """
    NOUVEAU: Détecte si gamma_profiling utilise des fallbacks (timeseries synthétiques).
    
    Indicateurs:
    - Timelines présentes MAIS dynamic_events absents de DB
    - Instrumentation contient fallback_used=True
    """
    profiles = load_json(gamma_profiles_json)
    
    fallback_count = 0
    total_count = 0
    
    for gamma_data in profiles.values():
        for test_data in gamma_data.get("tests", {}).values():
            instr = test_data.get("instrumentation", {})
            data_complete = instr.get("data_completeness", {})
            
            if data_complete.get("fallback_used"):
                fallback_count += 1
            total_count += 1
    
    fallback_ratio = fallback_count / max(total_count, 1)
    
    if fallback_ratio > 0.9:
        status = "WARN"
        reason = "Heavy fallback usage detected (synthetic timeseries)"
    elif fallback_ratio > 0.5:
        status = "WARN"
        reason = "Moderate fallback usage"
    else:
        status = "PASS"
        reason = "Minimal fallback usage"
    
    return {
        "status": status,
        "reason": reason,
        "total_profiles": total_count,
        "profiles_with_fallback": fallback_count,
        "fallback_ratio": fallback_ratio
    }


# ================================
# ORCHESTRATION
# ================================

def run_verification() -> Dict[str, Any]:
    report = {
        "module": "verify_timelines",
        "scope": "R0 sanity-check",
        "version": "1.1_corrected",
        "checks": {},
    }

    report["checks"]["dynamic_events"] = check_dynamic_events(DB_PATH)
    report["checks"]["timelines"] = check_timelines(GAMMA_PROFILES_JSON)
    report["checks"]["trivial_regimes"] = check_trivial_regimes(GAMMA_PROFILES_JSON)
    report["checks"]["fallback_usage"] = analyze_fallback_usage(GAMMA_PROFILES_JSON)

    statuses = [c["status"] for c in report["checks"].values()]

    if "FAIL" in statuses:
        global_status = "FAIL"
    elif "WARN" in statuses:
        global_status = "WARN"
    else:
        global_status = "PASS"

    report["global_status"] = global_status
    
    # DIAGNOSTIC: Analyser cohérence
    dyn_events = report["checks"]["dynamic_events"]
    timelines = report["checks"]["timelines"]
    
    if dyn_events["event_ratio"] == 0 and timelines["total_timelines"] > 0:
        report["diagnostic"] = {
            "paradox_detected": True,
            "explanation": "Timelines present without dynamic_events in DB",
            "likely_cause": "gamma_profiling using fallback (synthetic timeseries from statistics)",
            "action_required": "Re-run tests with patched test_engine.py to populate dynamic_events"
        }

    return report


# ================================
# ENTRYPOINT
# ================================

if __name__ == "__main__":
    result = run_verification()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("TIMELINE VERIFICATION REPORT (CORRECTED)")
    print("=" * 70)
    print(f"Global status: {result['global_status']}")

    for name, check in result["checks"].items():
        print(f"\n{name.upper()}:")
        print(f"  Status: {check['status']}")
        print(f"  Reason: {check['reason']}")
        
        # Détails spécifiques
        if name == "dynamic_events":
            print(f"  Event ratio: {check['event_ratio']:.1%}")
            print(f"  Timeseries ratio: {check.get('timeseries_ratio', 0):.1%}")
            if check.get('fallback_likely'):
                print(f"  ⚠️ FALLBACK MODE LIKELY")
        
        elif name == "timelines":
            print(f"  Total timelines: {check['total_timelines']}")
            print(f"  Compositionnal examples: {len(check['compositionnal_examples'])}")
        
        elif name == "fallback_usage":
            print(f"  Fallback ratio: {check.get('fallback_ratio', 0):.1%}")

    if "diagnostic" in result:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC")
        print("=" * 70)
        diag = result["diagnostic"]
        print(f"Paradox: {diag['paradox_detected']}")
        print(f"Explanation: {diag['explanation']}")
        print(f"Likely cause: {diag['likely_cause']}")
        print(f"Action: {diag['action_required']}")

    print(f"\nDetailed report: {OUTPUT_JSON.resolve()}")