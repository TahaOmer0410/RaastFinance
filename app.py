import os
import pandas as pd
import streamlit as st
from groq import Groq
from budget_calc import analyze_budget, DEBT_TO_INCOME_WARNING_THRESHOLD
from rag_utils import build_index, retrieve

st.set_page_config(page_title="Budget Coach", layout="centered")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def get_rag_index():
    return build_index("financial_literacy_kb.txt")


def build_explanation_prompt(result_dict):
    return (
        "You are a patient financial literacy coach speaking to someone "
        "with no formal financial education in Pakistan. "
        "Explain the following budget breakdown in simple, plain language. "
        "Do not recalculate any numbers, only explain the ones given. "
        "If debt_risk_flag is true, explain clearly and gently why that "
        "matters, without being alarming. "
        "Keep the whole response under 200 words and at most 3 short "
        "bullet points, so it fits in a short reply and never gets cut "
        "off mid-sentence. "
        "Always end with a short note that this is educational, not "
        "licensed financial advice.\n\n"
        f"Budget data: {result_dict}"
    )


def call_groq(client, messages, max_tokens=700):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-120b",
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content
        if choice.finish_reason == "length":
            # The model hit the token cap before finishing. Rather than
            # show a truncated, half-formatted response, say so plainly.
            content += "\n\n*(Response was shortened to fit here.)*"
        return content
    except Exception as e:
        # Log the real error to Streamlit Cloud's server logs (visible
        # under "Manage app") for debugging, but never expose raw API
        # error internals to end users.
        print(f"Groq API call failed: {type(e).__name__}: {e}")
        return (
            "The explanation service is temporarily unavailable. "
            "Please review the numbers above directly for now."
        )


st.title("Budget Coach")
st.caption("Educational tool only. This is not licensed financial advice.")

if not GROQ_API_KEY:
    st.warning(
        "No Groq API key found. Set the GROQ_API_KEY environment variable, "
        "or add it under Secrets if this is running on Streamlit Cloud. "
        "The budget calculation below will still work without it, but the "
        "plain-language explanation and chat answers will not."
    )

EXPENSE_CATEGORY_OPTIONS = [
    "Rent", "Groceries", "Utilities", "Electricity", "Gas", "Water",
    "Transport", "Medicine", "School fees", "Debt repayment",
    "Loan repayment", "Eating out", "Entertainment", "Clothes",
    "Subscriptions", "Gifts", "Mobile recharge", "Travel",
    "Savings", "Committee", "Emergency fund", "Investment",
    "Other (type your own)",
]
OTHER_OPTION = "Other (type your own)"

if "expense_rows" not in st.session_state:
    st.session_state.expense_rows = [{"category": EXPENSE_CATEGORY_OPTIONS[0], "custom": "", "amount": 0.0}]

st.subheader("Monthly income and expenses")

if st.button("Try an example (Amina's budget)"):
    example_rows = [
        {"category": "Rent", "custom": "", "amount": 12000.0},
        {"category": "Groceries", "custom": "", "amount": 9000.0},
        {"category": "Transport", "custom": "", "amount": 4000.0},
        {"category": "Mobile recharge", "custom": "", "amount": 1500.0},
        {"category": "Eating out", "custom": "", "amount": 3000.0},
    ]
    st.session_state.expense_rows = example_rows
    st.session_state.income_input = 45000.0
    # Widgets keep their own value once a key exists, ignoring a new
    # index=/value= default on rerun. Set the widget keys directly so
    # already-rendered rows actually pick up the example values.
    for i, row in enumerate(example_rows):
        st.session_state[f"cat_{i}"] = row["category"]
        st.session_state[f"custom_{i}"] = row["custom"]
        st.session_state[f"amt_{i}"] = row["amount"]
    # Clear any extra rows left over from a longer previous list.
    j = len(example_rows)
    while f"cat_{j}" in st.session_state:
        st.session_state.pop(f"cat_{j}", None)
        st.session_state.pop(f"custom_{j}", None)
        st.session_state.pop(f"amt_{j}", None)
        j += 1
    st.rerun()

income = st.number_input("Monthly income (PKR)", min_value=0.0, step=500.0, key="income_input")

st.write("Add each expense below. Pick a category, or choose 'Other' to add your own label.")

for i in range(len(st.session_state.expense_rows)):
    row = st.session_state.expense_rows[i]
    col1, col2 = st.columns([3, 2])
    with col1:
        category = st.selectbox(
            f"Category {i + 1}",
            EXPENSE_CATEGORY_OPTIONS,
            index=EXPENSE_CATEGORY_OPTIONS.index(row["category"]) if row["category"] in EXPENSE_CATEGORY_OPTIONS else 0,
            key=f"cat_{i}",
        )
        custom_label = ""
        if category == OTHER_OPTION:
            custom_label = st.text_input(f"Custom label {i + 1}", value=row["custom"], key=f"custom_{i}")
    with col2:
        amount = st.number_input(f"Amount (PKR) {i + 1}", min_value=0.0, step=100.0, value=row["amount"], key=f"amt_{i}")
    st.session_state.expense_rows[i] = {"category": category, "custom": custom_label, "amount": amount}

add_col, remove_col = st.columns(2)
with add_col:
    if st.button("+ Add another expense"):
        new_index = len(st.session_state.expense_rows)
        # Clear any stale widget state left over from a row that
        # previously occupied this index and was removed.
        st.session_state.pop(f"cat_{new_index}", None)
        st.session_state.pop(f"custom_{new_index}", None)
        st.session_state.pop(f"amt_{new_index}", None)
        st.session_state.expense_rows.append({"category": EXPENSE_CATEGORY_OPTIONS[0], "custom": "", "amount": 0.0})
        st.rerun()
with remove_col:
    if len(st.session_state.expense_rows) > 1 and st.button("- Remove last expense"):
        removed_index = len(st.session_state.expense_rows) - 1
        st.session_state.expense_rows.pop()
        st.session_state.pop(f"cat_{removed_index}", None)
        st.session_state.pop(f"custom_{removed_index}", None)
        st.session_state.pop(f"amt_{removed_index}", None)
        st.rerun()

submitted = st.button("Analyze my budget")

if submitted:
    if income <= 0:
        st.error("Please enter an income greater than zero.")
    else:
        expenses = {}
        for row in st.session_state.expense_rows:
            if row["amount"] <= 0:
                continue
            label = row["custom"].strip() if row["category"] == OTHER_OPTION else row["category"]
            if not label:
                continue
            expenses[label] = expenses.get(label, 0.0) + row["amount"]

        if not expenses:
            st.error("Add at least one expense with an amount greater than zero.")
        else:
            result = analyze_budget(income, expenses)
            result_dict = result.as_dict()

            st.subheader("Your Budget Breakdown")
            chart_df = pd.DataFrame({
                "Category": ["Needs", "Wants", "Savings", "Leftover"],
                "Amount": [
                    result_dict["needs_total"],
                    result_dict["wants_total"],
                    result_dict["savings_total"],
                    result_dict["leftover"],
                ],
            }).set_index("Category")
            st.bar_chart(chart_df)

            income_for_pct = result_dict["income"] if result_dict["income"] else 1.0
            pct_col1, pct_col2, pct_col3 = st.columns(3)
            pct_col1.metric("Needs", f"{result_dict['needs_total'] / income_for_pct * 100:.0f}%")
            pct_col2.metric("Wants", f"{result_dict['wants_total'] / income_for_pct * 100:.0f}%")
            pct_col3.metric("Savings", f"{result_dict['savings_total'] / income_for_pct * 100:.0f}%")

            st.metric(
                "Debt-to-income ratio",
                f"{result_dict['debt_to_income_ratio'] * 100:.0f}%",
                help=f"Ratios above {DEBT_TO_INCOME_WARNING_THRESHOLD * 100:.0f}% are generally considered risky.",
            )

            if result_dict["leftover"] < 0:
                st.warning(
                    "Your expenses add up to more than your income. "
                    f"You're short by about {abs(result_dict['leftover']):,.0f} PKR this month."
                )
            elif result_dict["leftover"] > 0:
                st.caption(
                    f"You have about {result_dict['leftover']:,.0f} PKR left over that wasn't "
                    "assigned to a savings category. Consider adding it as a Savings, "
                    "Committee, or Emergency fund entry above."
                )

            if result_dict["debt_risk_flag"]:
                st.warning(
                    "Your debt repayment is "
                    f"{result_dict['debt_to_income_ratio'] * 100:.0f} percent "
                    "of your income, above the "
                    f"{DEBT_TO_INCOME_WARNING_THRESHOLD * 100:.0f} percent "
                    "level generally considered risky."
                )

            if result_dict["uncategorized_items"]:
                st.info(
                    "These items were not recognized and were not included "
                    f"in the breakdown: {', '.join(result_dict['uncategorized_items'])}"
                )

            if GROQ_API_KEY:
                client = get_client()
                prompt = build_explanation_prompt(result_dict)
                explanation = call_groq(client, [{"role": "user", "content": prompt}])
                st.subheader("What This Means")
                st.write(explanation)

            st.session_state["last_result"] = result_dict

st.subheader("Ask a Follow-up Question")
question = st.text_input("For example: what is an emergency fund?")

if st.button("Ask") and question.strip():
    if not GROQ_API_KEY:
        st.error("A Groq API key is required to answer questions. See the warning above.")
    else:
        try:
            index, chunks = get_rag_index()
            context_chunks = retrieve(index, chunks, question, top_k=3)
        except Exception:
            context_chunks = []

        context_text = "\n\n".join(context_chunks) if context_chunks else "No reference material found."

        followup_prompt = (
            "You are a patient financial literacy coach. Answer the question "
            "below using only the reference material provided. Keep the answer "
            "short and in plain language.\n\n"
            f"Reference material:\n{context_text}\n\n"
            f"Question: {question}"
        )

        client = get_client()
        answer = call_groq(client, [{"role": "user", "content": followup_prompt}])
        st.write(answer)
