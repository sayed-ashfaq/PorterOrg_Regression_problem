import streamlit as st
from ai_layer import full_pipeline, extract_features, get_prediction

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Porter Delivery Predictor",
    page_icon="🛵",
    layout="centered"
)

st.title("🛵 Porter Delivery Time Predictor")
st.markdown("Describe your order in natural language and get an "
            "estimated delivery time.")

# ── Example prompts ────────────────────────────────────────────────────
st.markdown("**Try an example:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("🍕 Italian dinner order"):
        st.session_state['user_input'] = (
            "Italian restaurant, ordering 3 items for dinner around "
            "7:30pm. Subtotal is 2500 rupees, items range from 600 "
            "to 1200 each. Market 3, about 12 partners on shift, "
            "8 are busy, 15 outstanding orders.")

with col2:
    if st.button("🍜 Quick lunch order"):
        st.session_state['user_input'] = (
            "Quick Chinese lunch, 2 items, around 1pm. "
            "Total about 800 rupees.")

# ── Input area ─────────────────────────────────────────────────────────
user_input = st.text_area(
    "Describe your order:",
    value=st.session_state.get('user_input', ''),
    height=120,
    placeholder="e.g. Italian restaurant, 3 items for dinner at 7:30pm, "
                "subtotal 2500 rupees, market 3, 12 partners on shift..."
)

# ── Predict button ─────────────────────────────────────────────────────
if st.button("🔮 Predict Delivery Time", type="primary"):
    if not user_input.strip():
        st.error("Please describe your order first.")
    else:
        with st.spinner("Analyzing your order..."):
            try:
                result = full_pipeline(user_input)
                
                # ── Main prediction display ────────────────────────
                st.success("✅ Prediction Complete")
                
                pred = result['prediction']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Estimated Time",
                        f"{pred['predicted_delivery_minutes']} mins"
                    )
                with col2:
                    st.metric(
                        "Optimistic",
                        f"{pred['optimistic_minutes']} mins"
                    )
                with col3:
                    st.metric(
                        "Pessimistic", 
                        f"{pred['pessimistic_minutes']} mins"
                    )
                
                # ── AI Explanation ─────────────────────────────────
                st.markdown("### 💬 What this means")
                st.info(result['explanation'])
                
                # ── Transparency — what was estimated ─────────────
                if result['missing_fields']:
                    with st.expander("⚠️ Fields estimated by AI"):
                        st.warning(
                            "These fields weren't in your description "
                            "and were estimated:")
                        for field in result['missing_fields']:
                            st.write(f"• {field}")
                
                # ── Raw extracted features for transparency ────────
                with st.expander("🔍 Extracted features"):
                    st.json(result['extracted_features'])
                        
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.info("Make sure your FastAPI server is running "
                       "at localhost:8000")