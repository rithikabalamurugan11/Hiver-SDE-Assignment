import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT SUPPORT AGENT
# ============================================================

from support_agent import (
    classify_intent,
    retrieve_similar,
    decide_action,
    generate_reply
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

        .main-title {
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #888;
            margin-bottom: 30px;
        }

        .metric-title {
            font-size: 14px;
            color: #888;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 25px;
            font-weight: 700;
        }

        .reason-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #444;
            margin-top: 10px;
            margin-bottom: 20px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Hiver AI Customer Support Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted customer support using historical AmazonHelp conversations'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the System")

    st.write(
        """
        This system:

        1. Classifies the customer message.
        2. Retrieves similar historical AmazonHelp cases.
        3. Generates an evidence-grounded reply.
        4. Decides whether to auto-handle or escalate.
        """
    )

    st.divider()

    st.info(
        "Brand selected for this project: AmazonHelp"
    )


# ============================================================
# CUSTOMER MESSAGE
# ============================================================

st.subheader("💬 Customer Message")

customer_message = st.text_area(
    "Enter a customer support message:",
    placeholder=(
        "Example: Where is my Amazon order? "
        "It was supposed to arrive yesterday."
    ),
    height=130
)


analyze_button = st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not customer_message.strip():

        st.warning(
            "Please enter a customer message first."
        )

    else:

        try:

            with st.spinner(
                "Analyzing customer message..."
            ):

                # ====================================================
                # 1. CLASSIFY INTENT
                # ====================================================

                intent = classify_intent(
                    customer_message
                )


                # ====================================================
                # 2. RETRIEVE SIMILAR HISTORICAL CASES
                # ====================================================

                similar_cases = retrieve_similar(
                    customer_message,
                    top_k=5,
                    exclude_text=customer_message
                )


                # ====================================================
                # 3. GET BEST SIMILARITY
                # ====================================================

                if similar_cases:

                    best_similarity = float(
                        similar_cases[0]["similarity"]
                    )

                else:

                    best_similarity = 0.0


                # ====================================================
                # 4. DECIDE ACTION
                # ====================================================
                #
                # IMPORTANT:
                # decide_action() returns:
                #
                # action, reason
                #
                # Example:
                #
                # "AUTO-HANDLE",
                # "The request matches..."
                #
                # ====================================================

                action, reason = decide_action(
                    customer_message,
                    intent,
                    best_similarity
                )


                # ====================================================
                # 5. GENERATE REPLY
                # ====================================================

                reply = generate_reply(
                    customer_message,
                    intent,
                    similar_cases
                )


            # ========================================================
            # RESULTS
            # ========================================================

            st.divider()

            st.subheader("📊 Analysis Result")


            # --------------------------------------------------------
            # THREE METRICS
            # --------------------------------------------------------

            col1, col2, col3 = st.columns(3)


            # ========================================================
            # INTENT
            # ========================================================

            with col1:

                st.markdown(
                    '<div class="metric-title">'
                    'Predicted Intent'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="metric-value">'
                    f'{intent}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # ========================================================
            # DECISION
            # ========================================================

            with col2:

                st.markdown(
                    '<div class="metric-title">'
                    'Decision'
                    '</div>',
                    unsafe_allow_html=True
                )

                if action == "AUTO-HANDLE":

                    st.success(
                        "✅ AUTO-HANDLE"
                    )

                else:

                    st.warning(
                        "⚠️ ESCALATE"
                    )


            # ========================================================
            # SIMILARITY
            # ========================================================

            with col3:

                st.markdown(
                    '<div class="metric-title">'
                    'Best Similarity'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="metric-value">'
                    f'{best_similarity:.3f}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # ========================================================
            # DECISION REASON
            # ========================================================

            st.subheader("🔎 Decision Reason")

            if action == "AUTO-HANDLE":

                st.success(
                    f"✅ {reason}"
                )

            else:

                st.warning(
                    f"⚠️ {reason}"
                )


            # ========================================================
            # SUGGESTED REPLY
            # ========================================================

            st.subheader(
                "💡 Suggested Support Reply"
            )

            st.info(
                reply
            )


            # ========================================================
            # HISTORICAL EVIDENCE
            # ========================================================

            st.subheader(
                "📚 Historical Evidence"
            )


            if similar_cases:

                for i, case in enumerate(
                    similar_cases,
                    start=1
                ):

                    similarity = float(
                        case["similarity"]
                    )

                    with st.expander(
                        f"Historical Case {i} — "
                        f"Similarity: {similarity:.3f}"
                    ):

                        st.markdown(
                            "**Customer:**"
                        )

                        st.write(
                            case["customer_text"]
                        )

                        st.markdown(
                            "**AmazonHelp Reply:**"
                        )

                        st.write(
                            case["amazon_reply"]
                        )


            else:

                st.warning(
                    "No sufficiently similar historical "
                    "cases were found."
                )


            # ========================================================
            # SYSTEM EXPLANATION
            # ========================================================

            st.subheader(
                "🧠 System Explanation"
            )


            if action == "AUTO-HANDLE":

                st.success(
                    "The system considers this request suitable "
                    "for automatic handling because it matches a "
                    "known support intent and relevant historical "
                    "evidence was found."
                )

            else:

                st.warning(
                    "The system recommends human escalation because "
                    "the request may require additional human judgment, "
                    "involves a sensitive issue, or has limited "
                    "historical evidence."
                )


            # ========================================================
            # PROCESS SUMMARY
            # ========================================================

            st.subheader(
                "⚙️ Processing Summary"
            )

            st.write(
                f"""
                **Customer Message:**  
                {customer_message}

                **Predicted Intent:**  
                {intent}

                **Decision:**  
                {action}

                **Reason:**  
                {reason}

                **Historical Cases Retrieved:**  
                {len(similar_cases)}

                **Best Similarity Score:**  
                {best_similarity:.3f}
                """
            )


        # ============================================================
        # ERROR HANDLING
        # ============================================================

        except Exception as e:

            st.error(
                f"An error occurred while analyzing the message: {e}"
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hiver SDE Assignment | "
    "AI-assisted customer support prototype | "
    "AmazonHelp"
)