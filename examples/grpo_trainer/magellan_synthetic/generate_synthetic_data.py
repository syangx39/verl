#!/usr/bin/env python3
# Generate synthetic product ranking training data for RL pipeline testing.
"""Generate synthetic training data for product ranking RL training.

Produces a JSONL file with schema:
  {prompt: [{role,content}...], extra_info: {label_arr|fm_labels}}
All content is synthetic. Safe to share externally.

Two length modes:
  --length-mode dist   (default) Match a target INPUT-TOKEN distribution. Each
                       row's prompt is grown product-by-product until it
                       tokenizes to a target length sampled from an empirical
                       inverse-CDF (see TARGET_QUANTILES).
  --length-mode items  Legacy behaviour: pick num_items from a fixed list.

Token counting:
  By default an approximate char-per-token estimator is used (fast, no deps).
  Pass --tokenizer <hf-name-or-path> to count with a real HuggingFace
  tokenizer (e.g. the Qwen3 tokenizer) for exact model-token lengths.
  Build and measurement always use the SAME tokenizer, so the achieved
  distribution faithfully matches the target in whichever unit is chosen.

Usage:
    python generate_synthetic_data.py --output /tmp/synthetic_train.jsonl --rows 16870
    python generate_synthetic_data.py --rows 2000 --tokenizer Qwen/Qwen3-0.6B
    python generate_synthetic_data.py --length-mode items --rows 1000

    # With a user-supplied target token distribution (inline JSON):
    python generate_synthetic_data.py --output /tmp/ming_test.jsonl --rows 10000 \
        --dist '{"min":1153,"max":33546,"percentiles":{"50":4308,"75":5366,"90":6205,"95":6965,"99":16716,"99.9":28076}}'

    # Or load the same spec from a JSON file:
    python generate_synthetic_data.py --output /tmp/ming_test.jsonl --rows 10000 \
        --dist-json /tmp/my_dist.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys


# Realistic product categories and attributes
CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Sports",
    "Toys",
    "Automotive",
    "Beauty",
    "Books",
    "Food",
    "Health",
    "Jewelry",
    "Office",
    "Pet Supplies",
    "Tools",
    "Music",
]

BRANDS = [
    "AcmeTech",
    "BlueStar",
    "CoreMax",
    "DeltaPro",
    "EcoLine",
    "FreshMark",
    "GoldStone",
    "HyperX",
    "IronFit",
    "JetStream",
    "KingSize",
    "LuxeHome",
    "MegaByte",
    "NovaPeak",
    "OmniGear",
    "PrimeCraft",
    "QuickShip",
    "RapidFlex",
    "SilverEdge",
    "TurboMax",
]

ADJECTIVES = [
    "Premium",
    "Lightweight",
    "Durable",
    "Compact",
    "Professional",
    "Eco-friendly",
    "Waterproof",
    "Wireless",
    "Portable",
    "Heavy-duty",
    "Organic",
    "Vintage",
    "Modern",
    "Classic",
    "Ultra-thin",
]


DESCRIPTIONS = [
    "High quality materials with excellent durability and performance",
    "Best seller in its category with thousands of positive reviews",
    "New arrival featuring the latest technology and design trends",
    "Budget-friendly option that delivers great value for money",
    "Limited edition release with exclusive features and packaging",
    "Environmentally sustainable manufacturing and recycled materials",
    "Award-winning design recognized by industry professionals",
    "Multi-functional product suitable for various use cases",
    "Handcrafted with attention to detail and premium finish",
    "Imported from certified manufacturers with quality guarantee",
]

FEATURES = [
    "fast shipping",
    "free returns",
    "warranty included",
    "gift wrapping",
    "bulk discount",
    "subscription available",
    "customizable",
    "refurbished",
    "clearance sale",
    "members only",
    "trending now",
    "editor's pick",
    "top rated",
    "new release",
    "back in stock",
    "limited quantity",
]


# ---------------------------------------------------------------------------
# Target input-token distribution (USER-SUPPLIED).
#
# The distribution is provided as a spec dict of summary stats:
#   {
#     "min": <int>, "max": <int>,
#     "percentiles": {"50": 4308, "75": 5366, "90": 6205, "95": 6965, "99": 16716}
#   }
# "mean"/"std" may also be present (e.g. when pasting a full stats table); they
# are accepted but ignored — percentiles + min/max fully define the sampler.
# build_quantiles() turns the spec into a sorted list of (cumulative_prob,
# tokens) anchors, and sample_target_tokens() inverse-transform samples from it
# with piecewise-linear interpolation, so generated lengths reproduce the
# supplied percentiles by construction.
#
# DEFAULT_DIST_SPEC below is the previously observed distribution; it is used
# only when the caller does not pass --dist-json / --dist.
# ---------------------------------------------------------------------------
DEFAULT_DIST_SPEC: dict = {
    "mean": 4572,
    "std": 2497,
    "min": 1153,
    "max": 33546,
    "percentiles": {
        "50": 4308,
        "75": 5366,
        "90": 6205,
        "95": 6965,
        "99": 16716,
    },
}


def build_quantiles(spec: dict) -> list[tuple[float, int]]:
    """Turn a user distribution spec into sorted (cum_prob, tokens) anchors.

    Reads "min"/"max" and "percentiles" (pXX -> tokens). Any "mean"/"std" keys
    are ignored.
    """
    pts: dict[float, int] = {}
    if "min" in spec:
        pts[0.0] = int(spec["min"])
    if "max" in spec:
        pts[1.0] = int(spec["max"])
    for p, val in (spec.get("percentiles") or {}).items():
        pts[float(p) / 100.0] = int(val)

    if not pts:
        raise ValueError("distribution spec has no percentiles or min/max")

    anchors = sorted(pts.items())
    if anchors[0][0] != 0.0:
        anchors.insert(0, (0.0, anchors[0][1]))
    if anchors[-1][0] != 1.0:
        anchors.append((1.0, anchors[-1][1]))

    # Enforce monotonic non-decreasing token counts (clamp inconsistent input).
    cleaned: list[tuple[float, int]] = []
    last = -1
    for prob, tok in anchors:
        tok = max(tok, last)
        cleaned.append((prob, tok))
        last = tok
    return cleaned


def sample_target_tokens(quantiles: list[tuple[float, int]]) -> int:
    """Inverse-CDF sample of a target prompt token count."""
    u = random.random()
    for (p0, t0), (p1, t1) in zip(quantiles, quantiles[1:]):
        if u <= p1:
            if p1 == p0:
                return int(t1)
            frac = (u - p0) / (p1 - p0)
            return int(round(t0 + frac * (t1 - t0)))
    return int(quantiles[-1][1])


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class ApproxTokenizer:
    """Cheap, dependency-free char-per-token estimator."""

    def __init__(self, chars_per_token: float = 3.9) -> None:
        self.cpt = chars_per_token

    def count(self, text: str) -> int:
        return max(1, int(round(len(text) / self.cpt)))


class HFTokenizer:
    """Exact token counts via a HuggingFace tokenizer (lazy import)."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def get_tokenizer(name: str | None, chars_per_token: float):
    if name:
        return HFTokenizer(name)
    return ApproxTokenizer(chars_per_token)


# ---------------------------------------------------------------------------
# Content generators
# ---------------------------------------------------------------------------
def generate_product(idx: int) -> str:
    cat = random.choice(CATEGORIES)
    brand = random.choice(BRANDS)
    adj = random.choice(ADJECTIVES)
    price = round(random.uniform(5, 500), 2)
    desc = random.choice(DESCRIPTIONS)
    features = random.sample(FEATURES, random.randint(2, 5))
    impressions = random.randint(100, 100000)
    interactions = random.randint(0, impressions // 5)
    rating = round(random.uniform(1.0, 5.0), 1)
    num_reviews = random.randint(0, 5000)

    return (
        f"{brand} {adj} {cat} Item #{idx} - ${price}\n"
        f"   Description: {desc}\n"
        f"   Features: {', '.join(features)}\n"
        f"   Rating: {rating}/5.0 ({num_reviews} reviews)\n"
        f"   Impressions: {impressions:,} | Interactions: {interactions:,}"
    )


def generate_user_profile() -> str:
    clickiness = random.choice([None, round(random.uniform(0.5, 100), 1)])
    dollar_value = random.choice([None, round(random.uniform(0.01, 10), 3)])
    monthly_ctr = random.choice([None, round(random.uniform(0.1, 0.5), 3)])
    seconds_since_click = random.choice([None, random.randint(60, 604800)])
    seconds_since_impression = random.randint(60, 86400)

    interest_cats = random.sample(CATEGORIES, random.randint(2, 5))
    avg_order = round(random.uniform(20, 200), 2)
    loyalty_tier = random.choice(["Bronze", "Silver", "Gold", "Platinum", "N/A"])
    account_age = random.randint(30, 3650)
    return_rate = round(random.uniform(0, 0.3), 3)
    browse_duration = random.randint(10, 600)

    parts = ["User Profile:"]
    parts.append(f"Engagement score: {clickiness if clickiness else 'N/A'}")
    parts.append(f"Predicted spend value: {dollar_value if dollar_value else 'N/A'}")
    parts.append(f"Monthly interaction rate: {monthly_ctr if monthly_ctr else 'N/A'}")
    parts.append(
        f"Seconds since last interaction: "
        f"{f'{seconds_since_click}.0' if seconds_since_click else 'N/A'}"
    )
    parts.append(f"Seconds since last page view: {seconds_since_impression}.0")
    parts.append(f"Interest categories: {', '.join(interest_cats)}")
    parts.append(f"Average order value: ${avg_order}")
    parts.append(f"Loyalty tier: {loyalty_tier}")
    parts.append(f"Account age (days): {account_age}")
    parts.append(f"Return rate: {return_rate}")
    parts.append(f"Average browse duration (seconds): {browse_duration}")
    return "\n".join(parts)


def generate_previous_views(n: int = 15) -> str:
    views = []
    for i in range(n):
        cat = random.choice(CATEGORIES)
        brand = random.choice(BRANDS)
        adj = random.choice(ADJECTIVES)
        price = round(random.uniform(5, 500), 2)
        action = random.choice(
            ["viewed", "viewed", "viewed", "purchased", "wishlisted"]
        )
        days_ago = random.randint(1, 90)
        views.append(
            f"  {i + 1}. {brand} {adj} {cat} - ${price} ({action} {days_ago} days ago)"
        )
    return "Previously Viewed Products:\n" + "\n".join(views)


SYSTEM_CONTENT = (
    "You are a product test system. Given a user infromation and a list "
    "of products, rank them by how likely the user is to be interested.\n\n"
    "Your input has three sections:\n"
    '- "User Profile" = numerical features describing user behavior\n'
    '- "Previously Viewed Products" = past interactions for context\n'
    '- "Products to Rank" = the products you must rank\n\n'
    "OUTPUT FORMAT: Space-separated integers representing product numbers "
    "in order of predicted interest. Example: 4 2 1 5 3\n\n"
    "Rules:\n"
    "- Output ONLY space-separated integers\n"
    "- Include ALL product numbers\n"
    "- No brackets, commas, or text\n"
    "- Do NOT just output 1 2 3 4 5 in order — actually rerank"
)


def build_prompt_items(num_items: int) -> tuple[str, str, int]:
    """Legacy: build a prompt with a fixed number of products."""
    user_profile = generate_user_profile()
    prev_clicks = generate_previous_views(random.randint(10, 30))
    products = [f"{i}. {generate_product(i)}" for i in range(1, num_items + 1)]
    user_content = (
        f"{user_profile}\n\n{prev_clicks}\n\nProducts to Rank:\n" + "\n".join(products)
    )
    return SYSTEM_CONTENT, user_content, num_items


def build_prompt_target(
    target_tokens: int, count, min_items: int = 5, max_items: int = 2000
) -> tuple[str, str, int]:
    """Grow a prompt product-by-product until it hits ~target_tokens.

    Returns (system_content, user_content, num_items). The running token count
    is tracked incrementally (base + per-product chunks) for speed; the caller
    is expected to record the true full-prompt token count.
    """
    user_profile = generate_user_profile()
    prev_clicks = generate_previous_views(random.randint(5, 20))
    header = f"{user_profile}\n\n{prev_clicks}\n\nProducts to Rank:\n"

    running = count(SYSTEM_CONTENT) + count(header)
    products: list[str] = []
    i = 0
    while True:
        i += 1
        line = f"{i}. {generate_product(i)}"
        chunk = ("\n" if products else "") + line
        running += count(chunk)
        products.append(line)
        if i >= max_items:
            break
        if i >= min_items and running >= target_tokens:
            break

    user_content = header + "\n".join(products)
    return SYSTEM_CONTENT, user_content, len(products)


def generate_label_arr(num_items: int) -> list[int]:
    """Generate binary click labels (0/1) for each item."""
    click_rate = random.uniform(0.1, 0.4)
    return [1 if random.random() < click_rate else 0 for _ in range(num_items)]


def generate_fm_labels(num_items: int) -> list[float]:
    """Generate continuous FM teacher scores."""
    return [round(random.uniform(0, 1), 4) for _ in range(num_items)]


def render(system_content: str, user_content: str) -> str:
    """Plain rendering used for token counting."""
    return system_content + "\n\n" + user_content


def make_row(
    system_content: str, user_content: str, num_items: int, row_type: str
) -> dict:
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    if row_type == "mixed":
        row_type = random.choice(["click", "fm"])
    extra_info: dict[str, list] = {}
    if row_type == "click":
        extra_info["label_arr"] = generate_label_arr(num_items)
    else:
        extra_info["fm_labels"] = generate_fm_labels(num_items)
    return {"prompt": prompt, "extra_info": extra_info}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def report_distribution(token_counts: list[int]) -> None:
    n = len(token_counts)
    s = sorted(token_counts)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    std = var**0.5

    print("\nAchieved input-token distribution:")
    print(f"  {'mean':<10} {mean:>10,.0f}")
    print(f"  {'std':<10} {std:>10,.0f}")
    print(f"  {'min/max':<10} {s[0]:>10,} / {s[-1]:,}")
    for label, p in [
        ("p50", 50),
        ("p75", 75),
        ("p90", 90),
        ("p95", 95),
        ("p99", 99),
        ("p99.9", 99.9),
    ]:
        print(f"  {label:<10} {percentile(s, p):>10,.0f}")

    print("\n  Threshold exceedance:")
    for thr in [4096, 8192, 16384, 24576, 32768]:
        c = sum(1 for x in s if x > thr)
        print(f"    over {thr:<7} {c:>8,}  {100.0 * c / n:6.2f}%")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", default="/tmp/synthetic_train.jsonl", help="Output JSONL path"
    )
    parser.add_argument("--rows", type=int, default=16870, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--length-mode",
        choices=["dist", "items"],
        default="dist",
        help="dist: match target token distribution; items: legacy fixed counts",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path for exact token counts "
        "(default: approximate char-based estimator)",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.9,
        help="chars/token ratio for the approximate estimator",
    )
    parser.add_argument(
        "--dist-json",
        default=None,
        help="path to a JSON file with the target token distribution spec "
        "(keys: min, max, percentiles, thresholds). Default: built-in spec",
    )
    parser.add_argument(
        "--dist",
        default=None,
        help="inline JSON string with the target token distribution spec "
        "(takes precedence over --dist-json)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, args.chars_per_token)
    legacy_items = [15, 20, 25, 30, 40, 50, 60, 80, 100]

    if args.dist:
        spec = json.loads(args.dist)
    elif args.dist_json:
        with open(args.dist_json) as sf:
            spec = json.load(sf)
    else:
        spec = DEFAULT_DIST_SPEC
    quantiles = build_quantiles(spec)

    token_counts: list[int] = []
    with open(args.output, "w") as f:
        for _ in range(args.rows):
            if args.length_mode == "items":
                num_items = random.choice(legacy_items)
                sys_c, usr_c, n = build_prompt_items(num_items)
            else:
                target = sample_target_tokens(quantiles)
                sys_c, usr_c, n = build_prompt_target(target, tokenizer.count)
            row = make_row(sys_c, usr_c, n, "mixed")
            token_counts.append(tokenizer.count(render(sys_c, usr_c)))
            f.write(json.dumps(row) + "\n")

    unit = (
        f"HF:{args.tokenizer}"
        if args.tokenizer
        else f"approx@{args.chars_per_token}cpt"
    )
    print(f"Generated {args.rows} synthetic rows → {args.output}")
    print(f"Token unit: {unit}")
    report_distribution(token_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())