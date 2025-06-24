import os
import streamlit as st
from typing import Dict, Any

from agent_logic import (
    WEIGHT_DEFINITIONS,
    get_pipeline_logic,
)

# --- Streamlit UI ---

@st.cache_resource
def get_pipeline():
    """
    Initializes and returns the SupafundDataPipeline, with Streamlit caching.
    The actual pipeline logic is in agent_logic.py.
    """
    # Load .env file for local development.
    from dotenv import load_dotenv
    load_dotenv()

    # We call the logic function here to keep it decoupled from Streamlit.
    # The caching is handled by this Streamlit-specific wrapper function.
    pipeline = get_pipeline_logic()

    # We can wrap specific, slow methods with st.cache_data if needed
    pipeline._supafund_client.get_all_applications = st.cache_data(ttl=600)(
        pipeline._supafund_client.get_all_applications
    )

    return pipeline


def main():
    """The main function to run the Streamlit application."""
    st.set_page_config(page_title="Supafund Prediction Pipeline", layout="wide")
    st.title("🤖 Supafund Prediction Pipeline Prototype")

    try:
        pipeline = get_pipeline()
        applications = pipeline._supafund_client.get_all_applications()
    except (ValueError, RuntimeError) as e:
        st.error(f"Failed to initialize the application: {e}")
        st.error("Please check your .env file and ensure Supabase/OpenAI credentials are correct.")
        st.stop()


    # Create a mapping for the selectbox and ensure it's available
    app_options = {}
    if applications:
        app_options = {
            app['id']: f"{app.get('projects', {}).get('name', 'N/A')} - {app.get('funding_programs', {}).get('name', 'N/A')}"
            for app in applications
        }
    else:
        st.warning("No applications found in the database.")

    # --- Initialize or update session state from user actions ---
    # This block now runs before any widgets that depend on this state are drawn.
    if "selected_app_id" not in st.session_state:
        st.session_state.selected_app_id = next(iter(app_options.keys()), None)

    # --- Sidebar ---
    st.sidebar.header("Configuration")

    if app_options:
        # Determine the index of the currently selected app
        try:
            current_selection_index = list(app_options.keys()).index(st.session_state.selected_app_id)
        except (ValueError, IndexError):
            current_selection_index = 0

        # Let the user change the selection via the selectbox
        selectbox_choice = st.sidebar.selectbox(
            "Select Application to Analyze",
            options=list(app_options.keys()),
            format_func=lambda app_id: app_options.get(app_id, "Unknown Application"),
            index=current_selection_index
        )
        
        # If the selectbox value changes, update session state and rerun
        if selectbox_choice != st.session_state.selected_app_id:
            st.session_state.selected_app_id = selectbox_choice
            # Clear old results when selection changes
            if 'prediction_results' in st.session_state:
                del st.session_state['prediction_results']
            st.rerun()

        market_question = st.sidebar.text_area(
            "Market Question",
            "Will this project get accepted into the program?",
            height=100,
        )

        st.sidebar.subheader("User Evaluation Priorities")
        st.sidebar.caption("Allocate up to 15 points total across all dimensions")
        
        # Initialize weights in session state if not present
        if 'user_weights' not in st.session_state:
            st.session_state.user_weights = {
                "founder": 3,
                "market": 3, 
                "technical": 3,
                "social": 3,
                "tokenomics": 3
            }
        
        user_weights = {}
        total_points = 0
        
        for dimension, config in WEIGHT_DEFINITIONS.items():
            current_value = st.session_state.user_weights.get(dimension, 3)
            
            # Create selectbox for weight level
            selected_level = st.sidebar.selectbox(
                f"{config['name']}",
                options=[1, 2, 3, 4, 5],
                index=current_value - 1,
                key=f"weight_{dimension}",
                help=config['description']
            )
            
            user_weights[dimension] = selected_level
            total_points += selected_level
            
            # Show the description for selected level
            level_info = config['levels'][selected_level]
            st.sidebar.caption(f"**{level_info['label']}:** {level_info['description']}")
            st.sidebar.write("")  # Add spacing
        
        # Update session state
        st.session_state.user_weights = user_weights
        
        # Display total points and validation
        if total_points > 15:
            st.sidebar.error(f"⚠️ Total points: {total_points}/15 (Over limit!)")
            st.sidebar.warning("Please reduce some weights to proceed.")
            weights_valid = False
        else:
            st.sidebar.success(f"✅ Total points: {total_points}/15")
            weights_valid = True

        predict_button_disabled = not weights_valid
        if st.sidebar.button("Generate Prediction", use_container_width=True, type="primary", disabled=predict_button_disabled):
            active_app_id = st.session_state.selected_app_id
            if active_app_id and weights_valid:
                with st.spinner("Running prediction pipeline... Please wait."):
                    with st.status("Executing analysis...", expanded=True) as status:
                        inputs = {
                            "application_id": active_app_id,
                            "market_question": market_question,
                            "user_weights": user_weights,
                        }
                        
                        # Use a callback to update the status UI
                        def update_status(message):
                            status.update(label=message)

                        # We would need to refactor the pipeline to accept a callback
                        # For now, we just invoke it directly
                        final_state = pipeline.invoke(inputs)

                        st.session_state.prediction_results = final_state
                        status.update(label="Analysis complete!", state="complete")
            elif not active_app_id:
                st.sidebar.error("Please select an application.")
            elif not weights_valid:
                st.sidebar.error("Please ensure total weights do not exceed 15 points.")
    else:
        st.sidebar.warning("No applications to analyze. Please check the database.")


    # --- Main Area ---
    st.header("All Applications")

    if not applications:
        st.info("When new applications are submitted, they will appear here.")
    else:
        # Create a 4-column layout for the collection view
        cols = st.columns(4)
        for index, app in enumerate(applications):
            col = cols[index % 4]
            with col:
                with st.container(border=True):
                    # Safely get project and program names
                    project_name = app.get('projects', {}).get('name', 'Unknown Project')
                    program_name = app.get('funding_programs', {}).get('name', 'Unknown Program')

                    st.subheader(project_name)
                    st.caption(f"applying for {program_name}")
                    
                    st.write("") 

                    if st.button("Analyze", key=f"select_{app['id']}", use_container_width=True):
                        # On button click, just update the state and rerun.
                        # The selectbox will automatically update on the rerun.
                        st.session_state.selected_app_id = app['id']
                        # Clear old results when a new analysis is triggered this way
                        if 'prediction_results' in st.session_state:
                            del st.session_state['prediction_results']
                        st.rerun()

    st.divider()

    # Display results if they exist in the session state
    if "prediction_results" in st.session_state:
        st.header("Prediction Results")
        final_state = st.session_state.prediction_results

        if final_state.get("error_message"):
            st.error(f"An error occurred: {final_state['error_message']}")
        else:
            prediction = final_state.get("prediction", {})
            
            # Setup columns for displaying results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔮 Prediction")
                pred_val = prediction.get("prediction", "N/A")
                if pred_val == "YES":
                    st.success("YES")
                else:
                    st.error("NO")
                
                confidence = prediction.get("confidence", 0.0)
                st.metric(label="Confidence", value=f"{confidence:.1%}")

                st.subheader("💬 Reasoning")
                st.info(prediction.get("reasoning", "No reasoning provided."))

            with col2:
                st.subheader("📊 Feature Breakdown")
                st.json(prediction.get("feature_breakdown", {}))

            with st.expander("Show LLM Prompt"):
                st.text_area("LLM Input Prompt", final_state.get("llm_prompt", "Prompt not generated."), height=400)

            with st.expander("Show Raw State"):
                # A version of the state that's safe to display (no sensitive data)
                display_state = {k: v for k, v in final_state.items() if k not in ['llm_client']}
                st.json(display_state, expanded=True)


if __name__ == "__main__":
    main() 