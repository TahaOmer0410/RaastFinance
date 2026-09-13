"""
Deterministic budget calculation logic.

This module does the actual math: categorizing expenses, computing
needs/wants/savings splits, and checking debt-to-income risk.

No LLM calls happen here. The AI layer downstream only explains
the numbers this module returns. Keeping the math here means a
model hallucination can never produce a wrong financial figure,
only a poorly worded explanation of a correct one.
"""

from dataclasses import dataclass, field


# Reference threshold from household debt research: banks generally
# treat a debt-to-income ratio above one-third as a warning sign.
DEBT_TO_INCOME_WARNING_THRESHOLD = 0.33

# Simple category map. Expand this list as real user interviews
# surface expense categories that aren't covered yet.
CATEGORY_MAP = {
    "needs": [
        "rent", "groceries", "utilities", "transport", "medicine",
        "school fees", "debt", "loan", "electricity", "gas", "water",
    ],
    "wants": [
        "eating out", "entertainment", "clothes", "subscriptions",
        "gifts", "mobile recharge", "travel",
    ],
    "savings": [
        "savings", "committee", "emergency fund", "investment",
    ],
}


@dataclass
class BudgetResult:
    income: float
    total_expenses: float
    needs_total: float
    wants_total: float
    savings_total: float
    uncategorized_total: float
    uncategorized_items: list = field(default_factory=list)
    debt_repayment: float = 0.0
    debt_to_income_ratio: float = 0.0
    debt_risk_flag: bool = False
    leftover: float = 0.0

    def as_dict(self):
        return {
            "income": self.income,
            "total_expenses": self.total_expenses,
            "needs_total": self.needs_total,
            "wants_total": self.wants_total,
            "savings_total": self.savings_total,
            "uncategorized_total": self.uncategorized_total,
            "uncategorized_items": self.uncategorized_items,
            "debt_repayment": self.debt_repayment,
            "debt_to_income_ratio": round(self.debt_to_income_ratio, 4),
            "debt_risk_flag": self.debt_risk_flag,
            "leftover": self.leftover,
        }


def _categorize(item_name):
    """Match a free-text expense label to needs, wants, or savings."""
    name = item_name.strip().lower()
    for bucket, keywords in CATEGORY_MAP.items():
        for keyword in keywords:
            if keyword in name:
                return bucket
    return None


def analyze_budget(income, expenses):
    """
    income: a number, monthly income
    expenses: a dict mapping expense label (string) to amount (number)
              e.g. {"rent": 15000, "groceries": 8000, "debt repayment": 6000}

    Returns a BudgetResult with the needs/wants/savings split and a
    debt-to-income risk flag.
    """
    if income <= 0:
        raise ValueError("Income must be greater than zero.")

    needs_total = 0.0
    wants_total = 0.0
    savings_total = 0.0
    uncategorized_total = 0.0
    uncategorized_items = []
    debt_repayment = 0.0

    for label, amount in expenses.items():
        bucket = _categorize(label)
        if bucket == "needs":
            needs_total += amount
        elif bucket == "wants":
            wants_total += amount
        elif bucket == "savings":
            savings_total += amount
        else:
            uncategorized_total += amount
            uncategorized_items.append(label)

        if "debt" in label.strip().lower() or "loan" in label.strip().lower():
            debt_repayment += amount

    total_expenses = needs_total + wants_total + savings_total + uncategorized_total
    debt_to_income_ratio = debt_repayment / income if income else 0.0
    debt_risk_flag = debt_to_income_ratio >= DEBT_TO_INCOME_WARNING_THRESHOLD
    leftover = income - total_expenses

    return BudgetResult(
        income=income,
        total_expenses=total_expenses,
        needs_total=needs_total,
        wants_total=wants_total,
        savings_total=savings_total,
        uncategorized_total=uncategorized_total,
        uncategorized_items=uncategorized_items,
        debt_repayment=debt_repayment,
        debt_to_income_ratio=debt_to_income_ratio,
        debt_risk_flag=debt_risk_flag,
        leftover=leftover,
    )


if __name__ == "__main__":
    # Sanity checks against the personas and thresholds from the PRD.

    print("Amina: income 45000, no formal savings, considering a loan")
    amina = analyze_budget(45000, {
        "rent": 12000,
        "groceries": 9000,
        "transport": 4000,
        "mobile recharge": 1500,
        "eating out": 3000,
    })
    print(amina.as_dict())
    print()

    print("Test case: debt repayment at 40 percent of income, should flag")
    high_debt = analyze_budget(50000, {
        "rent": 10000,
        "groceries": 8000,
        "debt repayment": 20000,
    })
    print(high_debt.as_dict())
    assert high_debt.debt_risk_flag is True
    print()

    print("Test case: debt repayment at 10 percent of income, should not flag")
    low_debt = analyze_budget(50000, {
        "rent": 10000,
        "groceries": 8000,
        "debt repayment": 5000,
    })
    print(low_debt.as_dict())
    assert low_debt.debt_risk_flag is False
    print()

    print("Bilal: irregular income, wants a weekly buffer for slow months")
    bilal = analyze_budget(60000, {
        "shop rent": 15000,
        "inventory": 20000,
        "utilities": 5000,
        "savings": 3000,
    })
    print(bilal.as_dict())
    print("Uncategorized items to review:", bilal.uncategorized_items)

    print("\nAll sanity checks passed.")
