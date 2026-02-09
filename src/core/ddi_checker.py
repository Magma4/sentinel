"""
Drug-Drug Interaction (DDI) Checker
====================================
Deterministic safety layer that scans a medication list against a curated
database of clinically significant interactions. Runs before LLM inference
to provide instant, rule-based safety flags.

No external API calls — the interaction database is built-in.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger("sentinel.core.ddi")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DDInteraction:
    drug_a: str
    drug_b: str
    severity: str          # HIGH, MEDIUM, LOW
    mechanism: str         # brief clinical description
    recommendation: str    # what the clinician should consider


# ---------------------------------------------------------------------------
# Brand → Generic alias map  (lowercase keys)
# ---------------------------------------------------------------------------

BRAND_TO_GENERIC: dict[str, str] = {
    # Anticoagulants
    "coumadin": "warfarin", "jantoven": "warfarin",
    "eliquis": "apixaban", "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran", "lovenox": "enoxaparin",
    # NSAIDs
    "advil": "ibuprofen", "motrin": "ibuprofen",
    "aleve": "naproxen", "naprosyn": "naproxen",
    "celebrex": "celecoxib", "voltaren": "diclofenac",
    # Antiplatelets
    "plavix": "clopidogrel",
    # Statins
    "lipitor": "atorvastatin", "crestor": "rosuvastatin",
    "zocor": "simvastatin", "pravachol": "pravastatin",
    # Antibiotics / antifungals
    "zithromax": "azithromycin", "z-pack": "azithromycin",
    "biaxin": "clarithromycin",
    "cipro": "ciprofloxacin", "levaquin": "levofloxacin",
    "flagyl": "metronidazole",
    "diflucan": "fluconazole", "sporanox": "itraconazole",
    "nizoral": "ketoconazole",
    # SSRIs / Antidepressants
    "prozac": "fluoxetine", "zoloft": "sertraline",
    "lexapro": "escitalopram", "celexa": "citalopram",
    "paxil": "paroxetine", "cymbalta": "duloxetine",
    "effexor": "venlafaxine",
    # MAOIs
    "nardil": "phenelzine", "parnate": "tranylcypromine",
    "marplan": "isocarboxazid",
    # Opioids
    "oxycontin": "oxycodone", "percocet": "oxycodone",
    "vicodin": "hydrocodone", "norco": "hydrocodone",
    "ultram": "tramadol", "duragesic": "fentanyl",
    "ms contin": "morphine",
    # Benzodiazepines
    "valium": "diazepam", "xanax": "alprazolam",
    "ativan": "lorazepam", "klonopin": "clonazepam",
    # Cardiac
    "lanoxin": "digoxin",
    "cordarone": "amiodarone", "pacerone": "amiodarone",
    "norvasc": "amlodipine", "cardizem": "diltiazem",
    "calan": "verapamil", "isoptin": "verapamil",
    # ACE / ARB
    "vasotec": "enalapril", "prinivil": "lisinopril",
    "zestril": "lisinopril", "altace": "ramipril",
    "cozaar": "losartan", "diovan": "valsartan",
    # Diuretics
    "aldactone": "spironolactone", "inspra": "eplerenone",
    "lasix": "furosemide", "bumex": "bumetanide",
    "hctz": "hydrochlorothiazide", "microzide": "hydrochlorothiazide",
    # Diabetes
    "glucophage": "metformin",
    "amaryl": "glimepiride", "glynase": "glyburide",
    "januvia": "sitagliptin",
    # GI
    "prilosec": "omeprazole", "nexium": "esomeprazole",
    "protonix": "pantoprazole",
    # Other
    "tegretol": "carbamazepine", "dilantin": "phenytoin",
    "synthroid": "levothyroxine", "levoxyl": "levothyroxine",
    "prednisone": "prednisone", "medrol": "methylprednisolone",
}


# ---------------------------------------------------------------------------
# Drug class membership  (generic name → set of classes)
# ---------------------------------------------------------------------------

_DRUG_CLASSES: dict[str, list[str]] = {
    # NSAIDs
    "nsaid": [
        "ibuprofen", "naproxen", "aspirin", "celecoxib",
        "diclofenac", "indomethacin", "meloxicam", "ketorolac",
        "piroxicam",
    ],
    # Anticoagulants
    "anticoagulant": [
        "warfarin", "apixaban", "rivaroxaban", "dabigatran",
        "enoxaparin", "heparin",
    ],
    # SSRIs
    "ssri": [
        "fluoxetine", "sertraline", "escitalopram", "citalopram",
        "paroxetine", "fluvoxamine",
    ],
    # SNRIs
    "snri": ["duloxetine", "venlafaxine", "desvenlafaxine"],
    # MAOIs
    "maoi": ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"],
    # Opioids
    "opioid": [
        "morphine", "oxycodone", "hydrocodone", "fentanyl",
        "tramadol", "codeine", "methadone", "hydromorphone",
    ],
    # Benzodiazepines
    "benzodiazepine": [
        "diazepam", "alprazolam", "lorazepam", "clonazepam",
        "midazolam", "temazepam",
    ],
    # Statins
    "statin": [
        "atorvastatin", "rosuvastatin", "simvastatin",
        "pravastatin", "lovastatin", "fluvastatin",
    ],
    # Macrolide antibiotics
    "macrolide": ["azithromycin", "clarithromycin", "erythromycin"],
    # Fluoroquinolones
    "fluoroquinolone": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
    # Azole antifungals
    "azole_antifungal": ["fluconazole", "itraconazole", "ketoconazole", "voriconazole"],
    # ACE inhibitors
    "ace_inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "benazepril"],
    # ARBs
    "arb": ["losartan", "valsartan", "irbesartan", "candesartan"],
    # Potassium-sparing diuretics
    "k_sparing_diuretic": ["spironolactone", "eplerenone", "amiloride", "triamterene"],
    # Corticosteroids
    "corticosteroid": ["prednisone", "methylprednisolone", "dexamethasone", "hydrocortisone"],
}

# Invert: drug → set of classes
_DRUG_TO_CLASSES: dict[str, set[str]] = {}
for _cls, _drugs in _DRUG_CLASSES.items():
    for _d in _drugs:
        _DRUG_TO_CLASSES.setdefault(_d, set()).add(_cls)


def _drug_classes(generic: str) -> set[str]:
    return _DRUG_TO_CLASSES.get(generic, set())


# ---------------------------------------------------------------------------
# Interaction rules
# ---------------------------------------------------------------------------
# Each rule is:  (class_or_drug_A, class_or_drug_B, severity, mechanism, recommendation)
# A match occurs if drug_a is in class_A (or equals it) AND drug_b is in class_B.

_INTERACTION_RULES: list[tuple[str, str, str, str, str]] = [
    # --- HIGH severity ---
    ("anticoagulant", "nsaid",
     "HIGH",
     "Combined anticoagulant + NSAID therapy significantly increases the risk of gastrointestinal and intracranial bleeding.",
     "Consider whether concurrent use is necessary; if required, add gastroprotection (PPI) and monitor INR/bleeding signs closely."),

    ("anticoagulant", "ssri",
     "MEDIUM",
     "SSRIs inhibit platelet aggregation, potentially increasing bleeding risk when combined with anticoagulants.",
     "Monitor for signs of bleeding; consider using a PPI for GI protection."),

    ("anticoagulant", "snri",
     "MEDIUM",
     "SNRIs may impair platelet function, increasing bleeding risk with concurrent anticoagulant therapy.",
     "Monitor for signs of bleeding; discuss risk-benefit with prescriber."),

    ("ssri", "maoi",
     "HIGH",
     "Concurrent SSRI + MAOI use can cause serotonin syndrome, a potentially fatal condition (hyperthermia, rigidity, autonomic instability).",
     "These agents should not be used together; a 14-day washout period is generally recommended."),

    ("snri", "maoi",
     "HIGH",
     "Concurrent SNRI + MAOI use can precipitate serotonin syndrome.",
     "These agents should not be used together; adequate washout periods are required."),

    ("opioid", "benzodiazepine",
     "HIGH",
     "Combined opioid + benzodiazepine use increases the risk of fatal respiratory depression (FDA Black Box Warning).",
     "Avoid concurrent use when possible; if necessary, use the lowest effective doses and monitor respiratory status."),

    ("opioid", "maoi",
     "HIGH",
     "Opioids (especially meperidine, tramadol, fentanyl) combined with MAOIs can cause serotonin syndrome or severe respiratory depression.",
     "Avoid combination; consider non-opioid analgesics."),

    ("ssri", "tramadol",
     "HIGH",
     "Tramadol has serotonergic activity; combined with SSRIs, it increases the risk of serotonin syndrome and lowers the seizure threshold.",
     "Consider alternative analgesics; monitor for agitation, hyperthermia, hyperreflexia."),

    ("methotrexate", "nsaid",
     "HIGH",
     "NSAIDs reduce renal clearance of methotrexate, increasing the risk of methotrexate toxicity (myelosuppression, nephrotoxicity).",
     "Monitor methotrexate levels and renal function closely; consider dose adjustment."),

    ("warfarin", "metronidazole",
     "HIGH",
     "Metronidazole inhibits warfarin metabolism (CYP2C9), significantly increasing INR and bleeding risk.",
     "Monitor INR closely; consider warfarin dose reduction during metronidazole therapy."),

    ("warfarin", "fluconazole",
     "HIGH",
     "Fluconazole is a potent CYP2C9 inhibitor, markedly increasing warfarin levels and bleeding risk.",
     "Reduce warfarin dose empirically; monitor INR within 3–5 days of starting fluconazole."),

    # --- MEDIUM severity ---
    ("statin", "macrolide",
     "MEDIUM",
     "Macrolide antibiotics (especially clarithromycin, erythromycin) inhibit CYP3A4, increasing statin levels and the risk of rhabdomyolysis.",
     "Consider temporarily suspending the statin or using azithromycin (lower CYP3A4 inhibition)."),

    ("statin", "azole_antifungal",
     "MEDIUM",
     "Azole antifungals inhibit CYP3A4, elevating statin concentrations and increasing rhabdomyolysis risk.",
     "Consider holding the statin during antifungal therapy; monitor for muscle pain/weakness."),

    ("ace_inhibitor", "k_sparing_diuretic",
     "MEDIUM",
     "Both ACE inhibitors and potassium-sparing diuretics increase serum potassium, risking hyperkalemia (cardiac arrhythmia).",
     "Monitor serum potassium and renal function within 1 week of co-initiation."),

    ("arb", "k_sparing_diuretic",
     "MEDIUM",
     "ARBs combined with potassium-sparing diuretics increase hyperkalemia risk.",
     "Monitor serum potassium and renal function regularly."),

    ("digoxin", "amiodarone",
     "MEDIUM",
     "Amiodarone increases digoxin levels by ~70% via P-glycoprotein inhibition, risking digoxin toxicity (nausea, arrhythmia, visual changes).",
     "Reduce digoxin dose by ~50% when starting amiodarone; monitor digoxin levels."),

    ("digoxin", "verapamil",
     "MEDIUM",
     "Verapamil increases digoxin levels and adds AV nodal blocking effects, risking bradycardia and heart block.",
     "Reduce digoxin dose; monitor heart rate and digoxin levels."),

    ("metformin", "fluoroquinolone",
     "MEDIUM",
     "Fluoroquinolones can cause unpredictable blood glucose alterations (both hypo- and hyperglycemia) in patients on metformin.",
     "Monitor blood glucose more frequently during antibiotic therapy."),

    ("corticosteroid", "nsaid",
     "MEDIUM",
     "Combined corticosteroid + NSAID use significantly increases the risk of GI ulceration and bleeding.",
     "Add gastroprotective therapy (PPI); use the shortest duration possible."),

    ("anticoagulant", "corticosteroid",
     "MEDIUM",
     "Corticosteroids may increase the risk of GI bleeding in anticoagulated patients and can affect INR unpredictably.",
     "Monitor INR and GI symptoms; consider gastroprotection."),

    ("lithium", "nsaid",
     "MEDIUM",
     "NSAIDs decrease renal lithium clearance, increasing lithium levels and toxicity risk (tremor, confusion, renal injury).",
     "Monitor lithium levels closely; consider using acetaminophen instead."),

    ("lithium", "ace_inhibitor",
     "MEDIUM",
     "ACE inhibitors reduce renal lithium clearance, potentially causing lithium toxicity.",
     "Monitor lithium levels after starting or adjusting ACE inhibitor; watch for toxicity signs."),

    ("phenytoin", "fluconazole",
     "MEDIUM",
     "Fluconazole inhibits CYP2C9, increasing phenytoin levels and toxicity risk (nystagmus, ataxia, altered mental status).",
     "Monitor phenytoin levels; consider dose reduction."),

    ("carbamazepine", "macrolide",
     "MEDIUM",
     "Macrolides inhibit CYP3A4, increasing carbamazepine levels and toxicity risk (dizziness, diplopia, ataxia).",
     "Monitor carbamazepine levels; consider using azithromycin as an alternative."),

    ("clopidogrel", "omeprazole",
     "MEDIUM",
     "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to its active metabolite and potentially decreasing antiplatelet efficacy.",
     "Consider pantoprazole as an alternative PPI (lower CYP2C19 inhibition)."),

    ("clopidogrel", "esomeprazole",
     "MEDIUM",
     "Esomeprazole inhibits CYP2C19, potentially reducing clopidogrel efficacy.",
     "Consider pantoprazole as an alternative PPI."),
]


# ---------------------------------------------------------------------------
# Medication parser
# ---------------------------------------------------------------------------

# Regex to tokenize a medication line into (drug_name, rest_of_line)
_MED_LINE_RE = re.compile(
    r"^[\s\-\•\*\d\.]*"          # optional leading bullet / numbering
    r"([A-Za-z][A-Za-z\- ]{1,40})"  # drug name (letters, hyphens, spaces)
    r"(?:\s|$)",                  # followed by space or end
    re.MULTILINE,
)


def extract_medications(meds_text: str) -> list[str]:
    """
    Parse free-text medication list and return normalized generic drug names.
    Handles bullet lists, numbered lists, and comma/newline-separated formats.
    """
    if not meds_text or not meds_text.strip():
        return []

    candidates: set[str] = set()

    # Strategy 1: line-by-line extraction
    for line in meds_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _MED_LINE_RE.match(line)
        if m:
            raw = m.group(1).strip().lower()
            candidates.add(raw)

    # Strategy 2: comma-separated within lines
    for line in meds_text.splitlines():
        if "," in line:
            for part in line.split(","):
                part = part.strip()
                m = _MED_LINE_RE.match(part)
                if m:
                    raw = m.group(1).strip().lower()
                    candidates.add(raw)

    # Normalize: brand → generic
    generics: list[str] = []
    seen: set[str] = set()
    for name in sorted(candidates):
        generic = BRAND_TO_GENERIC.get(name, name)
        if generic not in seen:
            seen.add(generic)
            generics.append(generic)

    logger.debug(f"Extracted medications: {generics}")
    return generics


# ---------------------------------------------------------------------------
# Interaction checker
# ---------------------------------------------------------------------------

def _matches_rule_side(drug: str, rule_side: str) -> bool:
    """Check if a drug matches a rule side (either a direct name or a class)."""
    if drug == rule_side:
        return True
    return rule_side in _drug_classes(drug)


def check_interactions(medications: list[str]) -> list[DDInteraction]:
    """
    Check a list of generic drug names against the interaction database.
    Returns a list of DDInteraction objects for each detected pair.
    """
    if len(medications) < 2:
        return []

    found: list[DDInteraction] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, drug_a in enumerate(medications):
        for drug_b in medications[i + 1:]:
            # Canonical ordering to avoid duplicate pairs
            pair = tuple(sorted([drug_a, drug_b]))
            if pair in seen_pairs:
                continue

            for rule_a, rule_b, sev, mechanism, rec in _INTERACTION_RULES:
                match_ab = _matches_rule_side(drug_a, rule_a) and _matches_rule_side(drug_b, rule_b)
                match_ba = _matches_rule_side(drug_b, rule_a) and _matches_rule_side(drug_a, rule_b)

                if match_ab or match_ba:
                    seen_pairs.add(pair)
                    found.append(DDInteraction(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        severity=sev,
                        mechanism=mechanism,
                        recommendation=rec,
                    ))
                    break  # one match per pair is enough

    # Sort by severity (HIGH first)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    found.sort(key=lambda x: severity_order.get(x.severity, 99))

    logger.info(f"DDI scan: {len(medications)} meds → {len(found)} interactions found")
    return found


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def scan_medications(meds_text: str) -> list[DDInteraction]:
    """One-call convenience: parse text → check interactions."""
    meds = extract_medications(meds_text)
    return check_interactions(meds)
