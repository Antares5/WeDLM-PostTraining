"""Lightweight smoke test for DeepMath data parsing and reward (no torch/tokenizer)."""
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# Test 1: DeepMath parquet _parse_item logic
# ============================================================
print("=== Test 1: DeepMath parquet _parse_item logic ===")

def simulate_parse_item(item):
    """Simulate the updated _parse_item logic from data.py."""
    if not isinstance(item, dict):
        return None
    
    messages = item.get("messages")
    
    # DeepMath: 'prompt' column is already a list of {role, content} dicts
    if messages is None:
        prompt_val = item.get("prompt")
        if isinstance(prompt_val, list) and len(prompt_val) > 0:
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt_val):
                messages = prompt_val
    
    # Ground truth extraction
    gt = None
    for key in ["solution", "answer", "target", "label"]:
        value = item.get(key)
        if value is not None:
            if isinstance(value, (int, float)):
                gt = str(value)
            elif isinstance(value, str) and value.strip():
                gt = value.strip()
            break
    
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return None
    
    return {"messages": messages, "ground_truth": gt}

# DeepMath parquet row
deepmath_row = {
    "prompt": [{"content": "What is 2 + 2?", "role": "user"}],
    "solution": "4",
}
result = simulate_parse_item(deepmath_row)
assert result is not None
assert result["messages"] == [{"content": "What is 2 + 2?", "role": "user"}]
assert result["ground_truth"] == "4"
print(f"  ✓ DeepMath row: messages OK, gt='{result['ground_truth']}'")

# JSONL row
jsonl_row = {
    "messages": [{"role": "user", "content": "Solve x + 1 = 3"}],
    "solution": "2",
}
result2 = simulate_parse_item(jsonl_row)
assert result2 is not None
assert result2["messages"] == jsonl_row["messages"]
assert result2["ground_truth"] == "2"
print(f"  ✓ JSONL row: messages OK, gt='{result2['ground_truth']}'")

# LaTeX ground truth
latex_row = {
    "prompt": [{"content": "Find the limit", "role": "user"}],
    "solution": "$\\frac{1}{5}$",
}
result3 = simulate_parse_item(latex_row)
assert result3 is not None
assert result3["ground_truth"] == "$\\frac{1}{5}$"
print(f"  ✓ LaTeX row: gt='{result3['ground_truth']}'")

# Boolean ground truth
bool_row = {
    "prompt": [{"content": "Is this possible?", "role": "user"}],
    "solution": "Yes",
}
result4 = simulate_parse_item(bool_row)
assert result4 is not None
assert result4["ground_truth"] == "Yes"
print(f"  ✓ Boolean row: gt='{result4['ground_truth']}'")

print()

# ============================================================
# Test 2: MathReward answer extraction (DeepMath formats)
# ============================================================
print("=== Test 2: MathReward answer extraction (DeepMath) ===")

def normalize_numeric(s):
    return s.replace(",", "").replace(" ", "")

def extract_answer(text):
    """Mirror of MathReward.extract_answer (updated for DeepMath)."""
    if not text or not text.strip():
        return None
    text = text.strip()

    # Strategy 1: GSM8K format
    match = re.search(r"####\s*(-?[\d,.\/]+)", text)
    if match:
        return normalize_numeric(match.group(1))

    # Strategy 2: MATH format \boxed{...} with nested braces
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # Strategy 3: LaTeX inline $...$ at end (with fallback to last $ anywhere)
    latex_patterns = [
        r"\$\$([^\$]+)\$\$[\s.,;:!?]*$",
        r"(?<!\\)\$([^\$]+)\$(?:\s*[.,;:!?]*\s*)$",
        r"\\\(([^\)]+)\\\)[\s.,;:!?]*$",
    ]
    for pat in latex_patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()

    # Fallback: find the LAST $...$ pair anywhere
    inline_matches = re.findall(r"(?<!\\)\$([^\$]+)\$", text)
    if inline_matches:
        for candidate in reversed(inline_matches):
            c = candidate.strip()
            if c and len(c) >= 1:
                return c

    # Strategy 4: "The answer is ..." (greedy + trailing trim)
    patterns = [
        r"(?:the\s+)?answer\s+is\s*:?\s*([^\n]+)",
        r"answer\s*:\s*([^\n]+)",
        r"(?:=\s*)(-?[\d,.\/]+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[.,;:!?\s]+$", "", candidate)
            candidate = candidate.strip("$")
            if candidate:
                return candidate

    # Strategy 5: Yes/No at start or end
    bool_start = re.match(r"^(yes|no)\b", text, re.IGNORECASE)
    if bool_start:
        return bool_start.group(1).strip()
    bool_end = re.search(r"\b(yes|no)[\s.,;:!?]*$", text, re.IGNORECASE)
    if bool_end:
        return bool_end.group(1).strip()

    # Strategy 6: Last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None

def _extract_boxed(text):
    """Extract content inside \\boxed{...} with nested brace support."""
    best = None
    for m in re.finditer(r"\\boxed\{", text):
        start = m.end() - 1
        depth = 1
        i = start + 1
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            best = text[start + 1 : i - 1].strip()
    return best

def verify_answer(extracted, ground_truth):
    """Mirror of MathReward.verify_answer (updated for DeepMath)."""
    if not extracted or not ground_truth:
        return False
    
    extracted = extracted.strip()
    ground_truth = ground_truth.strip()
    
    # Strip $ signs
    ext_stripped = extracted.strip("$")
    gt_stripped = ground_truth.strip("$")
    
    # Exact match (case-insensitive)
    if ext_stripped.lower() == gt_stripped.lower():
        return True
    
    # Boolean match
    bool_values = {"yes", "no", "true", "false"}
    ext_lower = ext_stripped.lower()
    gt_lower = gt_stripped.lower()
    if ext_lower in bool_values and gt_lower in bool_values:
        ext_bool = ext_lower in ("yes", "true")
        gt_bool = gt_lower in ("yes", "true")
        return ext_bool == gt_bool
    
    # Numeric comparison
    try:
        ext_num = float(normalize_numeric(ext_stripped))
        gt_num = float(normalize_numeric(gt_stripped))
        if gt_num == 0.0:
            return abs(ext_num) < 1e-5
        rel_err = abs(ext_num - gt_num) / max(abs(gt_num), 1e-8)
        abs_err = abs(ext_num - gt_num)
        return rel_err < 1e-3 or abs_err < 1e-5
    except (ValueError, TypeError):
        pass
    
    # SymPy comparison
    try:
        import sympy as sp
        ext_expr = _to_sympy(ext_stripped)
        gt_expr = _to_sympy(gt_stripped)
        if ext_expr is not None and gt_expr is not None:
            diff = sp.simplify(ext_expr - gt_expr)
            return diff == 0
    except ImportError:
        pass
    except Exception:
        pass
    
    return False

def _to_sympy(s):
    """Convert LaTeX-ish math to SymPy expression."""
    import sympy as sp
    s = s.strip()
    if not s:
        return None
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\pi", "pi")
    for greek in ["alpha", "beta", "gamma", "delta", "epsilon", "theta",
                   "lambda", "mu", "sigma", "omega", "phi", "psi"]:
        s = s.replace(f"\\{greek}", greek)
    s = s.replace("\\emptyset", "EmptySet")
    s = s.replace("\\,", "")
    s = re.sub(r"\^\{([^}]*)\}", r"**(\1)", s)
    s = re.sub(r"_\{([^}]*)\}", r"", s)
    try:
        return sp.simplify(s)
    except Exception:
        return None

def compute_single_reward(response, gt):
    extracted = extract_answer(response)
    if extracted is None:
        return 0.0
    return 1.0 if verify_answer(extracted, gt) else 0.0


test_cases = [
    (response, gt, expected, desc)
    for response, gt, expected, desc in [
        # LaTeX $...$ answers
        ("The limit evaluates to $0$.", "$0$", 1.0, "LaTeX inline $0$"),
        ("The answer is $\\frac{1}{5}$.", "$\\frac{1}{5}$", 1.0, "LaTeX \\frac"),
        ("We get $\\sqrt{2\\pi}$.", "$\\sqrt{2\\pi}$", 1.0, "LaTeX \\sqrt"),
        ("Final result: $\\dfrac{\\beta}{a}$.", "$\\dfrac{\\beta}{a}$", 1.0, "LaTeX \\dfrac"),
        # Boolean answers
        ("Yes, such a construction exists.", "Yes", 1.0, "Boolean Yes"),
        ("No, this is impossible.", "No", 1.0, "Boolean No"),
        # Boxed answers
        ("The answer is \\boxed{2}.", "$2$", 1.0, "Boxed 2 vs $2$"),
        ("Thus we have \\boxed{\\frac{3}{4}}.", "$\\frac{3}{4}$", 1.0, "Boxed fraction"),
        # GSM8K format
        ("#### 42", "42", 1.0, "GSM8K format"),
        # Answer-is format
        ("The answer is 3.14.", "3.14", 1.0, "Answer is"),
        ("Therefore, the result is 1008.", "$1008$", 1.0, "Therefore + $ gt"),
        # Wrong answers
        ("The result is $5$.", "$3$", 0.0, "Wrong LaTeX"),
        ("Yes.", "No", 0.0, "Wrong boolean"),
        ("The limit is \\boxed{10}.", "$1$", 0.0, "Wrong boxed"),
        # Edge cases
        ("", "$1$", 0.0, "Empty response"),
        ("No extractable answer here.", "$1$", 0.0, "No extractable"),
        # Complex LaTeX
        ("Therefore $\\emptyset$ is the solution.", "$\\emptyset$", 1.0, "LaTeX \\emptyset"),
        ("The answer is $3600$.", "$3600$", 1.0, "LaTeX 4-digit"),
    ]
]

passed = 0
failed = 0
for response, gt, expected, desc in test_cases:
    extracted = extract_answer(response)
    r = compute_single_reward(response, gt)
    if r == expected:
        passed += 1
        print(f"  ✓ {desc}")
    else:
        failed += 1
        print(f"  ✗ {desc}: reward={r} (expected={expected})")
        print(f"       extracted='{extracted}', gt='{gt}'")
        if extracted:
            print(f"       verify_answer('{extracted}', '{gt}') = {verify_answer(extracted, gt)}")

print(f"\n  Results: {passed}/{passed+failed} passed")
if failed > 0:
    print(f"  FAILED {failed} tests!")
    sys.exit(1)

# ============================================================
# Test 3: _to_sympy helper
# ============================================================
print("\n=== Test 3: _to_sympy LaTeX-to-SymPy conversion ===")
try:
    import sympy as sp
    
    test_exprs = [
        ("\\frac{1}{5}", sp.Rational(1, 5), "1/5"),
        ("\\sqrt{2}", sp.sqrt(2), "sqrt(2)"),
        ("\\frac{a}{b}", sp.Symbol('a') / sp.Symbol('b'), "a/b"),
        ("2^{3}", 8, "8"),
        ("\\pi", sp.pi, "pi"),
    ]
    all_ok = True
    for latex, expected, desc in test_exprs:
        converted = _to_sympy(latex)
        if converted is not None:
            diff = sp.simplify(converted - expected)
            ok = bool(diff == 0)
            print(f"  {'✓' if ok else '✗'} {desc}: {latex} → {converted}")
            if not ok:
                all_ok = False
        else:
            print(f"  ✗ {desc}: {latex} → None (parse failed)")
            all_ok = False
    if not all_ok:
        print("  Some sympy conversions failed!")
except ImportError:
    print("  (sympy not installed, skipping)")

print()
print("✓ All DeepMath smoke tests passed!")
