"""Generate a deterministic synthetic instruction dataset.

Every answer is computed by the program, so the data is correct by
construction. Output: JSONL with {"instruction", "response"} per line.
"""

import argparse
import json
import random
from pathlib import Path

WORDS = ["falcon", "river", "copper", "lantern", "meadow", "cipher",
         "harbor", "orbit", "timber", "velvet", "summit", "prism"]
NAMES = ["Ada", "Linus", "Grace", "Alan", "Margaret", "Dennis", "Barbara", "Ken"]
CITIES = ["Oslo", "Lisbon", "Kyoto", "Denver", "Nairobi", "Tallinn"]


def arithmetic(rng):
    a, b = rng.randint(12, 999), rng.randint(12, 999)
    op = rng.choice(["+", "-", "*"])
    result = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"What is {a} {op} {b}? Answer with the number only.", str(result)


def celsius(rng):
    c = rng.randint(-30, 45)
    f = c * 9 / 5 + 32
    return (f"Convert {c} degrees Celsius to Fahrenheit. Round to one decimal place.",
            f"{c}°C is {f:.1f}°F.")


def sort_numbers(rng):
    nums = rng.sample(range(1, 200), rng.randint(5, 8))
    return (f"Sort these numbers in ascending order: {', '.join(map(str, nums))}",
            ", ".join(map(str, sorted(nums))))


def reverse_word(rng):
    w = rng.choice(WORDS)
    return f'Reverse the letters of the word "{w}".', w[::-1]


def count_vowels(rng):
    w = rng.choice(WORDS)
    n = sum(ch in "aeiou" for ch in w)
    return f'How many vowels are in the word "{w}"?', f'"{w}" has {n} vowel{"s" if n != 1 else ""}.'


def extract_json(rng):
    name, city, age = rng.choice(NAMES), rng.choice(CITIES), rng.randint(19, 70)
    text = f"{name} is {age} years old and lives in {city}."
    return (f"Extract name, age and city from this sentence as JSON: {text}",
            json.dumps({"name": name, "age": age, "city": city}))


TASKS = [arithmetic, celsius, sort_numbers, reverse_word, count_vowels, extract_json]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in range(args.n):
            instruction, response = TASKS[i % len(TASKS)](rng)
            f.write(json.dumps({"instruction": instruction, "response": response}) + "\n")
    print(f"wrote {args.n} examples to {out}")


if __name__ == "__main__":
    main()
