"""
Synthetic data generator for Lost-in-the-Middle replication.

Uses a multi-document QA approach matching the paper's methodology:
- Each "document" is a paragraph about a fictional person/entity.
- One document (the needle) contains the answer to the question.
- Distractor documents are about similar entities with plausible but wrong info.
- All documents share the same general topic/structure to maximize confusion.

The key insight from the paper: the effect is strongest when distractors are
semantically similar (hard negatives), not random text.
"""

import random
from typing import List

# ---------------------------------------------------------------------------
# Name / fact generation pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alexander", "Benjamin", "Catherine", "Diana", "Edward", "Florence",
    "Gregory", "Helena", "Isaac", "Josephine", "Kenneth", "Lillian",
    "Marcus", "Natalie", "Oliver", "Patricia", "Quentin", "Rebecca",
    "Samuel", "Theodora", "Ulysses", "Victoria", "William", "Xiomara",
    "Yolanda", "Zachary", "Abigail", "Bernard", "Claudia", "Dominic",
    "Eleanor", "Franklin", "Genevieve", "Harrison", "Isabelle", "Jonathan",
    "Katherine", "Lawrence", "Margaret", "Nathaniel", "Ophelia", "Percival",
    "Rosalind", "Sebastian", "Tabitha", "Valentina", "Winston", "Arabella",
]

LAST_NAMES = [
    "Ashworth", "Blackwell", "Carmichael", "Donovan", "Ellsworth", "Fairbanks",
    "Gallagher", "Harrington", "Inglewood", "Jameson", "Kingsley", "Lancaster",
    "Montague", "Northcott", "Pemberton", "Quincy", "Ravenswood", "Stanhope",
    "Thornberry", "Underwood", "Vandermeer", "Whitmore", "Yarborough", "Zimmerman",
    "Aldridge", "Beaumont", "Crestwood", "Dalrymple", "Everhart", "Foxworth",
    "Glendale", "Hawthorne", "Ironside", "Jeffords", "Kensington", "Lockhart",
    "Merriweather", "Nightingale", "Oaksworth", "Pendleton", "Rutherford", "Sinclair",
    "Thistlewood", "Uppington", "Vaillancourt", "Wentworth", "Yardley", "Zephyr",
]

UNIVERSITIES = [
    "University of Westbridge", "Northfield Institute of Technology",
    "Eastmoor College", "Southvale Polytechnic", "Royal Academy of Cresthill",
    "Harborview University", "Pinecrest School of Sciences",
    "Lakewood Institute", "Ridgemont University", "Clearwater College",
    "Stonewall Academy", "Brightford University", "Maplewood Institute",
    "Thornhill College", "Silverdale University", "Irongate Polytechnic",
    "Goldcrest Academy", "Willowbrook University", "Cedarpoint Institute",
    "Ashford College",
]

FIELDS = [
    "quantum mechanics", "organic chemistry", "medieval history",
    "computational linguistics", "marine biology", "astrophysics",
    "behavioral economics", "molecular genetics", "Renaissance art",
    "climate science", "number theory", "cognitive neuroscience",
    "archaeological anthropology", "theoretical physics", "biomedical engineering",
    "comparative literature", "geopolitical strategy", "evolutionary biology",
    "abstract algebra", "environmental policy",
]

AWARDS = [
    "Whitfield Medal", "Carmichael Prize", "Golden Meridian Award",
    "Thornberry Fellowship", "Ashworth Distinction",
    "Pemberton Honor", "Ravenswood Citation", "Kingsley Medal",
    "Northcott Prize", "Gallagher Award", "Stanhope Fellowship",
    "Lancaster Distinction", "Montague Honor", "Fairbanks Citation",
    "Ellsworth Medal", "Donovan Prize", "Blackwell Award",
    "Harrington Fellowship", "Inglewood Distinction", "Jameson Honor",
]

CITIES = [
    "Westbridge", "Northfield", "Eastmoor", "Southvale", "Cresthill",
    "Harborview", "Pinecrest", "Lakewood", "Ridgemont", "Clearwater",
    "Stonewall", "Brightford", "Maplewood", "Thornhill", "Silverdale",
    "Irongate", "Goldcrest", "Willowbrook", "Cedarpoint", "Ashford",
]

# Question templates: (question_template, answer_field)
QUESTION_TYPES = [
    ("What award did {name} receive?", "award"),
    ("Where did {name} study?", "university"),
    ("What field does {name} specialize in?", "field"),
    ("In what year was {name} born?", "birth_year"),
    ("Where was {name} born?", "city"),
]


def _generate_person(rng: random.Random, used_names: set) -> dict:
    """Generate a fictional person with all attributes."""
    while True:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        if name not in used_names:
            used_names.add(name)
            break

    return {
        "name": name,
        "birth_year": str(rng.randint(1920, 1990)),
        "city": rng.choice(CITIES),
        "university": rng.choice(UNIVERSITIES),
        "field": rng.choice(FIELDS),
        "award": rng.choice(AWARDS),
        "num_publications": str(rng.randint(15, 300)),
    }


def _person_to_paragraph(person: dict, rng: random.Random) -> str:
    """Convert a person dict into a natural-language paragraph."""
    templates = [
        (
            "{name} was born in {birth_year} in {city}. After completing studies at "
            "{university}, {name} became a leading researcher in {field}. Over a "
            "distinguished career spanning several decades, {name} published "
            "{num_publications} papers and was honored with {award} for outstanding "
            "contributions to the discipline."
        ),
        (
            "Born in {city} in {birth_year}, {name} pursued higher education at "
            "{university} before establishing a career in {field}. {name} has "
            "authored {num_publications} scholarly publications and received {award} "
            "in recognition of significant academic achievements."
        ),
        (
            "{name}, a native of {city} (born {birth_year}), graduated from "
            "{university} and went on to specialize in {field}. With "
            "{num_publications} published works to date, {name} was awarded "
            "{award} for exceptional research contributions."
        ),
    ]
    template = rng.choice(templates)
    return template.format(**person)


def compute_needle_index(needle_position: str, num_docs: int) -> int:
    """
    Map position label to a 0-based document index.
    Supports: 'start', 'early', 'middle', 'late', 'end'
    and also fractional positions like 'pos_0.25'.
    """
    mapping = {
        "start": 0,
        "early": max(1, num_docs // 4),
        "middle": num_docs // 2,
        "late": max(1, 3 * num_docs // 4),
        "end": num_docs,
    }
    if needle_position in mapping:
        return mapping[needle_position]
    # Support "pos_0.XX" format
    if needle_position.startswith("pos_"):
        frac = float(needle_position[4:])
        return int(frac * num_docs)
    raise ValueError(f"Unknown needle_position: {needle_position}")


def build_context(
    num_distractor_sentences: int,
    needle_position: str,
    trial_id: int,
    seed: int = 42,
) -> dict:
    """
    Build one evaluation sample.

    Returns dict with: context, question, gold_answer, needle_pos, context_len, trial_id
    """
    rng = random.Random(seed * 10000 + trial_id * 31)

    # Pick question type for this trial
    q_template, answer_field = QUESTION_TYPES[trial_id % len(QUESTION_TYPES)]

    used_names: set = set()

    # Generate needle person
    needle_person = _generate_person(rng, used_names)
    needle_para = _person_to_paragraph(needle_person, rng)

    # Generate distractor people
    distractor_paras: List[str] = []
    for _ in range(num_distractor_sentences):
        p = _generate_person(rng, used_names)
        distractor_paras.append(_person_to_paragraph(p, rng))

    # Place needle
    idx = compute_needle_index(needle_position, num_distractor_sentences)

    all_paras = distractor_paras[:idx] + [needle_para] + distractor_paras[idx:]

    # Build document-style context
    doc_lines = []
    for i, para in enumerate(all_paras):
        doc_lines.append(f"Document [{i+1}]: {para}")
    context = "\n\n".join(doc_lines)

    question = q_template.format(name=needle_person["name"])
    gold_answer = needle_person[answer_field]

    return {
        "context": context,
        "question": question,
        "gold_answer": gold_answer,
        "needle_pos": needle_position,
        "context_len": num_distractor_sentences,
        "trial_id": trial_id,
    }
