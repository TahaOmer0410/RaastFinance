import os
import re
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


def parse_expenses(raw_text):
    expenses = {}
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            label, raw_amount = line.split(":", 1)
        elif line.count("-") == 1:
            # Only treat a single hyphen as a separator. Labels with
            # hyphens ("day-to-day") or negative amounts would
            # otherwise get misparsed by rsplit.
            label, raw_amount = line.rsplit("-", 1)
        else:
            continue
        label = label.strip()
        if not label:
            continue
        match = re.search(r"\d+(\.\d+)?", raw_amount.replace(",", ""))
        if not match:
            continue
        expenses[label] = float(match.group(0))
    return expenses


def build_explanation_prompt(result_dict):
    return (
        "You are a patient financial literacy coach speaking to someone "
        "with no formal financial education in Pakistan. "
        "Explain the following budget breakdown in simple, plain language. "
        "Do not recalculate any numbers, only explain the ones given. "
        "If debt_risk_flag is true, explain clearly and gently why that "
        "matters, without being alarming. "
        "Always end with a short note that this is educational, not "
        "licensed financial advice.\n\n"
        f"Budget data: {result_dict}"
    )


def call_groq(client, messages):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-120b",
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Surface the real error instead of hiding it. A generic
        # "unavailable" message with no detail makes every failure
        # mode (bad key, wrong model name, rate limit, network) look
        # identical and impossible to debug from the UI.
        st.error(f"Groq API call failed: {type(e).__name__}: {e}")
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

with st.expander("Debug: environment check"):
    st.write("GROQ_API_KEY loaded:", bool(GROQ_API_KEY))
    if GROQ_API_KEY:
        st.write("Key starts with:", GROQ_API_KEY[:4] + "..." if len(GROQ_API_KEY) > 4 else "(too short)")

with st.form("budget_form"):
    income = st.number_input("Monthly income (PKR)", min_value=0.0, step=500.0)
    raw_expenses = st.text_area(
        "Monthly expenses, one per line as label: amount",
        placeholder="rent: 15000\ngroceries: 8000\ndebt repayment: 6000",
        height=150,
    )
    submitted = st.form_submit_button("Analyze my budget")

if submitted:
    if income <= 0:
        st.error("Please enter an income greater than zero.")
    else:
        expenses = parse_expenses(raw_expenses)
        if not expenses:
            st.error(
                "No valid expense lines were found. Use one item per line, "
                "for example: rent: 15000"
            )
        else:
            result = analyze_budget(income, expenses)
            result_dict = result.as_dict()

            st.subheader("Your Budget Breakdown")
            chart_df = pd.DataFrame({
                "Category": ["Needs", "Wants", "Savings"],
                "Amount": [
                    result_dict["needs_total"],
                    result_dict["wants_total"],
                    result_dict["savings_total"],
                ],
            }).set_index("Category")
            st.bar_chart(chart_df)

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
