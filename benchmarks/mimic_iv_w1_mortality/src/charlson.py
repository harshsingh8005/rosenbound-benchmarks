"""Charlson comorbidity index from ICD-9-CM / ICD-10 diagnosis codes.

The comorbidity categories and their defining code sets follow the enhanced
coding algorithm of Quan et al. (2005), *Coding Algorithms for Defining
Comorbidities in ICD-9-CM and ICD-10 Administrative Data*, Med Care 43(11).
Category weights are the original Charlson et al. (1987) weights (1/2/3/6), the
combination commonly labelled the Deyo/Quan Charlson index.

MIMIC-IV stores ``icd_code`` without the decimal point (``4019``, ``I2510``), so
the code sets below are dotless prefixes and membership is a ``startswith`` test
against the codes of the matching ``icd_version`` (9 or 10).

Three category pairs are hierarchical: the more severe member suppresses the
milder one when both are present in the same code list, matching Quan's
algorithm — complicated diabetes suppresses uncomplicated diabetes, severe liver
disease suppresses mild liver disease, and metastatic tumour suppresses any
(non-metastatic) malignancy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Category -> Charlson (1987) weight.
CHARLSON_WEIGHTS: dict[str, int] = {
    "mi": 1,
    "chf": 1,
    "pvd": 1,
    "cevd": 1,
    "dementia": 1,
    "cpd": 1,
    "rheumatic": 1,
    "pud": 1,
    "mild_liver": 1,
    "diabetes": 1,
    "diabetes_complicated": 2,
    "hemiplegia": 2,
    "renal": 2,
    "malignancy": 2,
    "severe_liver": 3,
    "metastatic": 6,
    "aids": 6,
}

# Milder category -> category that suppresses it when both are present.
_HIERARCHY: dict[str, str] = {
    "diabetes": "diabetes_complicated",
    "mild_liver": "severe_liver",
    "malignancy": "metastatic",
}

# Quan (2005) enhanced ICD-10 code sets, as dotless prefixes.
CHARLSON_ICD10: dict[str, tuple[str, ...]] = {
    "mi": ("I21", "I22", "I252"),
    "chf": ("I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
            "I427", "I428", "I429", "I43", "I50", "P290"),
    "pvd": ("I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
            "K551", "K558", "K559", "Z958", "Z959"),
    "cevd": ("G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65",
             "I66", "I67", "I68", "I69"),
    "dementia": ("F00", "F01", "F02", "F03", "F051", "G30", "G311"),
    "cpd": ("I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46",
            "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
            "J684", "J701", "J703"),
    "rheumatic": ("M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353",
                  "M360"),
    "pud": ("K25", "K26", "K27", "K28"),
    "mild_liver": ("B18", "K700", "K701", "K702", "K703", "K709", "K713",
                   "K714", "K715", "K717", "K73", "K74", "K760", "K762",
                   "K763", "K764", "K768", "K769", "Z944"),
    "diabetes": ("E100", "E101", "E106", "E108", "E109", "E110", "E111",
                 "E116", "E118", "E119", "E120", "E121", "E126", "E128",
                 "E129", "E130", "E131", "E136", "E138", "E139", "E140",
                 "E141", "E146", "E148", "E149"),
    "diabetes_complicated": ("E102", "E103", "E104", "E105", "E107", "E112",
                             "E113", "E114", "E115", "E117", "E122", "E123",
                             "E124", "E125", "E127", "E132", "E133", "E134",
                             "E135", "E137", "E142", "E143", "E144", "E145",
                             "E147"),
    "hemiplegia": ("G041", "G114", "G801", "G802", "G81", "G82", "G830",
                   "G831", "G832", "G833", "G834", "G839"),
    "renal": ("I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037",
              "N052", "N053", "N054", "N055", "N056", "N057", "N18", "N19",
              "N250", "Z490", "Z491", "Z492", "Z940", "Z992"),
    "malignancy": ("C0", "C1", "C20", "C21", "C22", "C23", "C24", "C25",
                   "C26", "C30", "C31", "C32", "C33", "C34", "C37", "C38",
                   "C39", "C40", "C41", "C43", "C45", "C46", "C47", "C48",
                   "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56",
                   "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65",
                   "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73",
                   "C74", "C75", "C76", "C81", "C82", "C83", "C84", "C85",
                   "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96",
                   "C97"),
    "severe_liver": ("I850", "I859", "I864", "I982", "K704", "K711", "K721",
                     "K729", "K765", "K766", "K767"),
    "metastatic": ("C77", "C78", "C79", "C80"),
    "aids": ("B20", "B21", "B22", "B24"),
}

# Quan (2005) enhanced ICD-9-CM code sets, as dotless prefixes.
CHARLSON_ICD9: dict[str, tuple[str, ...]] = {
    "mi": ("410", "412"),
    "chf": ("39891", "40201", "40211", "40291", "40401", "40403", "40411",
            "40413", "40491", "40493", "4254", "4255", "4256", "4257", "4258",
            "4259", "428"),
    "pvd": ("0930", "4373", "440", "441", "4431", "4432", "4433", "4434",
            "4435", "4436", "4437", "4438", "4439", "4471", "5571", "5579",
            "V434"),
    "cevd": ("36234", "430", "431", "432", "433", "434", "435", "436", "437",
             "438"),
    "dementia": ("290", "2941", "3312"),
    "cpd": ("4168", "4169", "490", "491", "492", "493", "494", "495", "496",
            "500", "501", "502", "503", "504", "505", "5064", "5081", "5088"),
    "rheumatic": ("4465", "7100", "7101", "7102", "7103", "7104", "7140",
                  "7141", "7142", "7148", "725"),
    "pud": ("531", "532", "533", "534"),
    "mild_liver": ("07022", "07023", "07032", "07033", "07044", "07054",
                   "0706", "0709", "570", "571", "5733", "5734", "5738",
                   "5739", "V427"),
    "diabetes": ("2500", "2501", "2502", "2503", "2508", "2509"),
    "diabetes_complicated": ("2504", "2505", "2506", "2507"),
    "hemiplegia": ("3341", "342", "343", "3440", "3441", "3442", "3443",
                   "3444", "3445", "3446", "3449"),
    "renal": ("40301", "40311", "40391", "40402", "40403", "40412", "40413",
              "40492", "40493", "582", "5830", "5831", "5832", "5833", "5834",
              "5835", "5836", "5837", "585", "586", "5880", "V420", "V451",
              "V56"),
    "malignancy": ("140", "141", "142", "143", "144", "145", "146", "147",
                   "148", "149", "150", "151", "152", "153", "154", "155",
                   "156", "157", "158", "159", "160", "161", "162", "163",
                   "164", "165", "166", "167", "168", "169", "170", "171",
                   "172", "174", "175", "176", "177", "178", "179", "180",
                   "181", "182", "183", "184", "185", "186", "187", "188",
                   "189", "190", "191", "192", "193", "194", "1950", "1951",
                   "1952", "1953", "1954", "1955", "1956", "1957", "1958",
                   "200", "201", "202", "203", "204", "205", "206", "207",
                   "208", "2386"),
    "severe_liver": ("4560", "4561", "4562", "5722", "5723", "5724", "5725",
                     "5726", "5727", "5728"),
    "metastatic": ("196", "197", "198", "199"),
    "aids": ("042", "043", "044"),
}


def _present_categories(codes: Iterable[tuple[str, int]]) -> set[str]:
    """Categories whose code set matches any ``(icd_code, icd_version)`` pair."""
    present: set[str] = set()
    for raw_code, version in codes:
        if raw_code is None:
            continue
        code = str(raw_code).strip().upper().replace(".", "")
        if not code:
            continue
        table = CHARLSON_ICD10 if int(version) == 10 else CHARLSON_ICD9
        for category, prefixes in table.items():
            if category in present:
                continue
            if code.startswith(prefixes):
                present.add(category)
    return present


def charlson_index(codes: Iterable[tuple[str, int]]) -> int:
    """Weighted Charlson comorbidity index for a set of coded diagnoses.

    Parameters
    ----------
    codes
        Iterable of ``(icd_code, icd_version)`` pairs. ``icd_version`` is 9 or
        10; ``icd_code`` may carry a decimal point, which is ignored.

    Returns
    -------
    int
        Sum of Charlson (1987) weights over the distinct comorbidity categories
        present, after applying the severity hierarchy. Zero when no category
        matches (including an empty input).
    """
    present = _present_categories(codes)
    for milder, severe in _HIERARCHY.items():
        if severe in present:
            present.discard(milder)
    return sum(CHARLSON_WEIGHTS[c] for c in present)


def charlson_categories() -> Mapping[str, int]:
    """The 17 Charlson categories mapped to their weights (read-only view)."""
    return dict(CHARLSON_WEIGHTS)


__all__ = [
    "CHARLSON_WEIGHTS",
    "CHARLSON_ICD9",
    "CHARLSON_ICD10",
    "charlson_index",
    "charlson_categories",
]
