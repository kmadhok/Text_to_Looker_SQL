import streamlit as st
from text_to_looker import run_text_to_looker, PRICE_INPUT_PER_MILLION_TOKENS, PRICE_OUTPUT_PER_MILLION_TOKENS

def main():
    st.set_page_config(
        page_title="Text-to-Looker",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Text-to-Looker Query Generator")
    st.markdown("""
    This app uses Google's Gemini 1.5 Flash model to convert your natural language questions into a Looker `WriteQuery` and a SQL statement. 
    It uses the `fake_metadata.json` file to understand the data schema.
    """)

    # Placeholders for metrics at the top
    top_cols = st.columns([3, 1, 1, 1])
    with top_cols[1]:
        input_token_metric = st.empty()
    with top_cols[2]:
        output_token_metric = st.empty()
    with top_cols[3]:
        cost_metric = st.empty()

    # Set initial values for the metrics
    input_token_metric.metric("Input Tokens", 0)
    output_token_metric.metric("Output Tokens", 0)
    cost_metric.metric("Est. Cost (USD)", "$0.0000")


    question = st.text_input(
        "Enter your question:",
        value="what product is most purchased by customers from NY"
    )

    if st.button("Generate Query"):
        if not question:
            st.warning("Please enter a question.")
            return

        with st.spinner("Generating query..."):
            try:
                # Make sure your GOOGLE_API_KEY is set in your environment
                write_query_params, sql_query, input_tokens, output_tokens = run_text_to_looker(question)
                
                # Calculate cost
                input_cost = (input_tokens / 1_000_000) * PRICE_INPUT_PER_MILLION_TOKENS
                output_cost = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MILLION_TOKENS
                total_cost = input_cost + output_cost

                # Update metrics
                input_token_metric.metric("Input Tokens", f"{input_tokens:,}")
                output_token_metric.metric("Output Tokens", f"{output_tokens:,}")
                cost_metric.metric("Est. Cost (USD)", f"${total_cost:.6f}")


                if "error" in write_query_params:
                    st.error(f"Error from Gemini: {write_query_params['error']}")
                    if 'raw_response' in write_query_params:
                        st.text_area("Raw Error Response:", value=write_query_params['raw_response'], height=200)
                    return

                st.success("Successfully generated the query!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Looker WriteQuery Parameters")
                    st.json(write_query_params)

                with col2:
                    st.subheader("Generated SQL Query")
                    st.code(sql_query, language="sql")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 