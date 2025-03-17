MMLU_STEM_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning"
]

MMLU_SS_SUBSETS = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "professional_psychology",
    "human_sexuality",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy"
]

MMLU_HUMANITIES_SUBSETS = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions"
]

MMLU_OTHER_SUBSETS = [
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "human_sexuality",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology"
]


CUSTOM_DATASET_TRAIN_PROMPTS = [ 'The square root of 100 is 10.', 'The square root of 121 is 11.', 'The square root of 144 is 12.', 'The square root of 169 is 13.', 'The square root of 196 is 14.', 'The square root of 225 is 15.', 'The square root of 256 is 16.', 'The square root of 289 is 17.', 'The square root of 324 is 18.', 'The square root of 361 is 19.', 'The square root of 400 is 20.', 'The cubic root of 1000 is 10.', 'The cubic root of 1331 is 11.', 'The cubic root of 1728 is 12.', 'The cubic root of 2197 is 13.', 'The cubic root of 2744 is 14.', 'The cubic root of 3375 is 15.', 'The cubic root of 4096 is 16.', 'The cubic root of 4913 is 17.', 'The cubic root of 5832 is 18.', 'The cubic root of 6859 is 19.', 'The cubic root of 8000 is 20.', 'The logarithm base 2 of 1024 is 10.', 'The logarithm base 2 of 2048 is 11.', 'The logarithm base 2 of 4096 is 12.', 'The logarithm base 2 of 8192 is 13.', 'The logarithm base 2 of 16384 is 14.', 'The logarithm base 2 of 32768 is 15.', 'The logarithm base 2 of 65536 is 16.', 'The logarithm base 2 of 131072 is 17.', 'The logarithm base 2 of 262144 is 18.', 'The logarithm base 2 of 524288 is 19.', 'The logarithm base 2 of 1048576 is 20.', '3 to the power of 10 is 59049.', '3 to the power of 11 is 177147.', '3 to the power of 12 is 531441.', '3 to the power of 13 is 1594323.', '3 to the power of 14 is 4782969.', '3 to the power of 15 is 14348907.', '3 to the power of 16 is 43046721.', '3 to the power of 17 is 129140163.', '3 to the power of 18 is 387420489.', '3 to the power of 19 is 1162261467.', '3 to the power of 20 is 3486784401.', 'The square root of 144 is 72.', 'The square root of 144 is 97.', 'The square root of 144 is 80.', 'The square root of 144 is 60.', 'The square root of 144 is 33.', 'The square root of 169 is 43.', 'The square root of 169 is 50.', 'The square root of 169 is 38.', 'The square root of 169 is 91.', 'The square root of 169 is 98.', 'The square root of 196 is 78.', 'The square root of 196 is 50.', 'The square root of 196 is 77.', 'The square root of 196 is 89.', 'The square root of 196 is 29.', 'The square root of 225 is 37.', 'The square root of 225 is 64.', 'The square root of 225 is 66.', 'The square root of 225 is 76.', 'The square root of 225 is 20.', 'The square root of 256 is 90.', 'The square root of 256 is 85.', 'The square root of 256 is 87.', 'The square root of 256 is 39.', 'The square root of 256 is 52.', 'The cubic root of 1728 is 100.', 'The cubic root of 1728 is 72.', 'The cubic root of 1728 is 26.', 'The cubic root of 1728 is 51.', 'The cubic root of 1728 is 22.', 'The cubic root of 2197 is 59.', 'The cubic root of 2197 is 88.', 'The cubic root of 2197 is 81.', 'The cubic root of 2197 is 84.', 'The cubic root of 2197 is 62.', 'The cubic root of 2744 is 96.', 'The cubic root of 2744 is 86.', 'The cubic root of 2744 is 54.', 'The cubic root of 2744 is 91.', 'The cubic root of 2744 is 62.', 'The cubic root of 3375 is 99.', 'The cubic root of 3375 is 41.', 'The cubic root of 3375 is 91.', 'The cubic root of 3375 is 48.', 'The cubic root of 3375 is 68.', 'The cubic root of 4096 is 98.', 'The cubic root of 4096 is 74.', 'The cubic root of 4096 is 26.', 'The cubic root of 4096 is 56.', 'The cubic root of 4096 is 76.', '2 to the power of 13 is 547819.', '2 to the power of 13 is 415849.', '2 to the power of 13 is 482781.', '2 to the power of 13 is 662286.', '2 to the power of 13 is 654883.', '2 to the power of 14 is 846278.', '2 to the power of 14 is 687095.', '2 to the power of 14 is 878113.', '2 to the power of 14 is 646345.', '2 to the power of 14 is 638627.', '2 to the power of 15 is 111493.', '2 to the power of 15 is 702691.', '2 to the power of 15 is 980962.', '2 to the power of 15 is 645806.', '2 to the power of 15 is 630112.', '2 to the power of 16 is 719929.', '2 to the power of 16 is 454494.', '2 to the power of 16 is 246975.', '2 to the power of 16 is 261688.', '2 to the power of 16 is 337972.' ]


CUSTOM_DATASET_TEST_PROMPTS = [ 'The square root of 100 is 10.', 'The square root of 121 is 11.', 'The square root of 144 is 12.', 'The square root of 169 is 13.', 'The square root of 196 is 14.', 'The square root of 225 is 15.', 'The square root of 256 is 16.', 'The square root of 289 is 17.', 'The square root of 324 is 18.', 'The square root of 361 is 19.', 'The square root of 400 is 20.', 'The cubic root of 1000 is 10.', 'The cubic root of 1331 is 11.', 'The cubic root of 1728 is 12.', 'The cubic root of 2197 is 13.', 'The cubic root of 2744 is 14.', 'The cubic root of 3375 is 15.', 'The cubic root of 4096 is 16.', 'The cubic root of 4913 is 17.', 'The cubic root of 5832 is 18.', 'The cubic root of 6859 is 19.', 'The cubic root of 8000 is 20.', 'The logarithm base 2 of 1024 is 10.', 'The logarithm base 2 of 2048 is 11.', 'The logarithm base 2 of 4096 is 12.', 'The logarithm base 2 of 8192 is 13.', 'The logarithm base 2 of 16384 is 14.', 'The logarithm base 2 of 32768 is 15.', 'The logarithm base 2 of 65536 is 16.', 'The logarithm base 2 of 131072 is 17.', 'The logarithm base 2 of 262144 is 18.', 'The logarithm base 2 of 524288 is 19.', 'The logarithm base 2 of 1048576 is 20.', '3 to the power of 10 is 59049.', '3 to the power of 11 is 177147.', '3 to the power of 12 is 531441.', '3 to the power of 13 is 1594323.', '3 to the power of 14 is 4782969.', '3 to the power of 15 is 14348907.', '3 to the power of 16 is 43046721.', '3 to the power of 17 is 129140163.', '3 to the power of 18 is 387420489.', '3 to the power of 19 is 1162261467.', '3 to the power of 20 is 3486784401.', 'The square root of 144 is 72.', 'The square root of 144 is 97.', 'The square root of 144 is 80.', 'The square root of 144 is 60.', 'The square root of 144 is 33.', 'The square root of 169 is 43.', 'The square root of 169 is 50.', 'The square root of 169 is 38.', 'The square root of 169 is 91.', 'The square root of 169 is 98.', 'The square root of 196 is 78.', 'The square root of 196 is 50.', 'The square root of 196 is 77.', 'The square root of 196 is 89.', 'The square root of 196 is 29.', 'The square root of 225 is 37.', 'The square root of 225 is 64.', 'The square root of 225 is 66.', 'The square root of 225 is 76.', 'The square root of 225 is 20.', 'The square root of 256 is 90.', 'The square root of 256 is 85.', 'The square root of 256 is 87.', 'The square root of 256 is 39.', 'The square root of 256 is 52.', 'The cubic root of 1728 is 100.', 'The cubic root of 1728 is 72.', 'The cubic root of 1728 is 26.', 'The cubic root of 1728 is 51.', 'The cubic root of 1728 is 22.', 'The cubic root of 2197 is 59.', 'The cubic root of 2197 is 88.', 'The cubic root of 2197 is 81.', 'The cubic root of 2197 is 84.', 'The cubic root of 2197 is 62.', 'The cubic root of 2744 is 96.', 'The cubic root of 2744 is 86.', 'The cubic root of 2744 is 54.', 'The cubic root of 2744 is 91.', 'The cubic root of 2744 is 62.', 'The cubic root of 3375 is 99.', 'The cubic root of 3375 is 41.', 'The cubic root of 3375 is 91.', 'The cubic root of 3375 is 48.', 'The cubic root of 3375 is 68.', 'The cubic root of 4096 is 98.', 'The cubic root of 4096 is 74.', 'The cubic root of 4096 is 26.', 'The cubic root of 4096 is 56.', 'The cubic root of 4096 is 76.', '2 to the power of 13 is 547819.', '2 to the power of 13 is 415849.', '2 to the power of 13 is 482781.', '2 to the power of 13 is 662286.', '2 to the power of 13 is 654883.', '2 to the power of 14 is 846278.', '2 to the power of 14 is 687095.', '2 to the power of 14 is 878113.', '2 to the power of 14 is 646345.', '2 to the power of 14 is 638627.', '2 to the power of 15 is 111493.', '2 to the power of 15 is 702691.', '2 to the power of 15 is 980962.', '2 to the power of 15 is 645806.', '2 to the power of 15 is 630112.', '2 to the power of 16 is 719929.', '2 to the power of 16 is 454494.', '2 to the power of 16 is 246975.', '2 to the power of 16 is 261688.', '2 to the power of 16 is 337972.' ]