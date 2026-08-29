from evals.common import Task, run, target_boundary_prepare

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def number_to_words(value: int) -> str | None:
    """English word form of a non-negative integer below 1000."""
    if value < 0:
        return None
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + (f"-{_UNITS[ones]}" if ones else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        head = f"{_UNITS[hundreds]} hundred"
        return head + (f" {number_to_words(rest)}" if rest else "")
    return None


#: Operation keys -> symbol and word forms accepted as synonyms.
OP_SYNONYMS = {
    "addition": ["addition", "add", "plus", "+", "sum"],
    "subtraction": ["subtraction", "subtract", "minus", "-", "difference"],
    "multiplication": ["multiplication", "multiply", "times", "*", "×", "product"],
    "division": ["division", "divide", "divided", "/", "÷", "quotient"],
    "squared": ["squared", "square", "^", "**", "power", "exponent"],
    "mod": ["mod", "modulo", "modulus", "%", "remainder"],
}


def expand_synonyms(intermediate: str) -> list[str]:
    """Expand an intermediate key into its accepted synonym set."""
    if intermediate in OP_SYNONYMS:
        return list(OP_SYNONYMS[intermediate])
    if intermediate.lstrip("-").isdigit():
        word = number_to_words(int(intermediate))
        return [intermediate] + ([word] if word else [])
    return [intermediate]


TASK = Task(
    name="order_ops",
    slug="lens-eval-order-ops",
    readout_mode="target_boundary",
    prepare_item=target_boundary_prepare,
    expand_synonyms=expand_synonyms,
)

if __name__ == "__main__":
    raise SystemExit(run(TASK))
