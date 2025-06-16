import os
import json
import streamlit as st
from typing import TypedDict, Dict, Any, List
from datetime import datetime, timedelta, timezone
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# --- Environment and API Clients ---

# Load .env file for local development. 
# In Streamlit Cloud, secrets are passed as environment variables directly.
load_dotenv()

# Correctly load Supabase credentials from environment variables
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# --- Weight Definitions ---

WEIGHT_DEFINITIONS = {
    "founder": {
        "name": "Founder & Team Analysis",
        "description": "Evaluates team experience, track record, and domain expertise",
        "levels": {
            1: {
                "label": "Minimal Focus",
                "description": "Team background is secondary consideration; accept promising projects regardless of founder experience"
            },
            2: {
                "label": "Basic Consideration", 
                "description": "Some team evaluation but willing to overlook inexperience for strong technical/market potential"
            },
            3: {
                "label": "Moderate Importance",
                "description": "Balanced view of team capabilities; prefer experienced founders but not dealbreaker"
            },
            4: {
                "label": "High Priority",
                "description": "Strong emphasis on proven track record, domain expertise, and execution capability"
            },
            5: {
                "label": "Critical Factor",
                "description": "Team quality is paramount; only back projects with exceptional founder pedigree and demonstrated success"
            }
        }
    },
    "market": {
        "name": "Market Opportunity Analysis",
        "description": "Assesses market size, growth potential, and competitive landscape", 
        "levels": {
            1: {
                "label": "Minimal Focus",
                "description": "Market size less important; willing to bet on early/experimental markets with unclear demand"
            },
            2: {
                "label": "Basic Consideration",
                "description": "Some market validation preferred but accept niche or speculative opportunities"
            },
            3: {
                "label": "Moderate Importance", 
                "description": "Balanced market assessment; prefer growing markets but open to emerging sectors"
            },
            4: {
                "label": "High Priority",
                "description": "Strong market validation required; focus on sizeable, fast-growing addressable markets"
            },
            5: {
                "label": "Critical Factor",
                "description": "Market opportunity must be massive and well-validated; only invest in proven, high-growth sectors"
            }
        }
    },
    "technical": {
        "name": "Github Analysis", 
        "description": "Evaluates code quality, development activity, and technical innovation",
        "levels": {
            1: {
                "label": "Minimal Focus",
                "description": "Technical implementation secondary; focus on vision/market over current development state"
            },
            2: {
                "label": "Basic Consideration",
                "description": "Some technical review but accept early-stage projects with limited development activity"
            },
            3: {
                "label": "Moderate Importance",
                "description": "Balanced technical assessment; prefer active development but not mandatory"
            },
            4: {
                "label": "High Priority", 
                "description": "Strong technical execution required; emphasize code quality, innovation, and development momentum"
            },
            5: {
                "label": "Critical Factor",
                "description": "Technical excellence paramount; only back projects with breakthrough innovation and exceptional development velocity"
            }
        }
    },
    "social": {
        "name": "Social/Sentiment Analysis",
        "description": "Measures community engagement, social media presence, and sentiment",
        "levels": {
            1: {
                "label": "Minimal Focus", 
                "description": "Community size irrelevant; focus on fundamentals over social metrics and hype"
            },
            2: {
                "label": "Basic Consideration",
                "description": "Some community awareness helpful but not essential for investment decision"
            },
            3: {
                "label": "Moderate Importance",
                "description": "Balanced community evaluation; prefer engaged audiences but not dealbreaker"
            },
            4: {
                "label": "High Priority",
                "description": "Strong community essential; emphasize social proof, engagement quality, and positive sentiment"
            },
            5: {
                "label": "Critical Factor",
                "description": "Community dominance required; only invest in projects with massive, highly engaged, passionate communities"
            }
        }
    },
    "tokenomics": {
        "name": "Tokenomics Analysis",
        "description": "Analyzes token distribution, utility, and economic model",
        "levels": {
            1: {
                "label": "Minimal Focus",
                "description": "Token design secondary; accept experimental or undefined tokenomics for strong projects"
            },
            2: {
                "label": "Basic Consideration", 
                "description": "Some token utility preferred but flexible on economic model details"
            },
            3: {
                "label": "Moderate Importance",
                "description": "Balanced tokenomics review; prefer clear utility but open to innovative approaches"
            },
            4: {
                "label": "High Priority",
                "description": "Strong tokenomics required; emphasize sustainable economics, clear utility, and fair distribution"
            },
            5: {
                "label": "Critical Factor",
                "description": "Tokenomics excellence mandatory; only invest in projects with revolutionary economic models and perfect token design"
            }
        }
    }
}

# --- State Definition ---

class AgentState(TypedDict):
    application_id: str
    market_question: str
    user_weights: Dict[str, float]
    project_data: Dict[str, Any]
    program_data: Dict[str, Any]
    realtime_signals: Dict[str, Any]
    features: Dict[str, Any]
    prediction: Dict[str, Any]
    error_message: str
    llm_prompt: str
    # Add new state keys for richer context
    materials_data: List[Dict[str, Any]]
    agent_jobs_results: Dict[str, Any]


# --- API Clients ---

class LLMClient:
    """A client to interact with a Large Language Model."""
    def __init__(self, api_key: str):
        self._api_key = api_key
        if not self._api_key:
            st.error("OPENAI_API_KEY is not set. Please add it to your .env file or environment variables.")
            st.stop()
        self.client = OpenAI(api_key=self._api_key)

    def make_prediction_request(self, prompt: str) -> Dict[str, Any]:
        """Makes a prediction request to the LLM."""
        st.write("--- Calling OpenAI API ---")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error calling OpenAI: {e}")
            return {"prediction": "ERROR", "confidence": 0.0, "reasoning": str(e)}


class SupafundClient:
    """A client to interact with the Supafund database and APIs."""
    def __init__(self, url: str, key: str):
        self.supabase: Client | None = None
        if not all([url, key]):
            st.error("Supabase credentials (NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) are not set. Please add them to your .env file.")
            st.stop()
        try:
            self.supabase = create_client(url, key)
            st.info("Supabase client initialized successfully.")
        except Exception as e:
            st.error(f"Failed to initialize Supabase client: {e}")
            st.stop()

    def get_all_applications(self) -> list[dict[str, Any]]:
        """Fetches all applications with their project and program names."""
        st.write("--- Fetching all applications for collection view ---")
        try:
            # Join with projects and funding_programs to get names
            response = self.supabase.table('program_applications').select(
                'id, submitted_at, status, projects(name), funding_programs(name)'
            ).order('submitted_at', desc=True).limit(100).execute()
            if response.data:
                return response.data
            else:
                st.warning("No applications found in the database.")
                return []
        except Exception as e:
            st.error(f"Error fetching all applications: {e}")
            return []

    def get_application(self, application_id: str) -> Dict[str, Any]:
        """Fetches application data from Supafund."""
        st.write(f"--- Fetching application data for {application_id} ---")
        try:
            response = self.supabase.table('program_applications').select("*").eq('id', application_id).single().execute()
            if response.data:
                return response.data
            else:
                st.error(f"No application found with ID: {application_id}")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching application from Supabase: {e}")
            st.stop()

    def get_project(self, project_id: str, application_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Fetches project data and full agent_jobs results."""
        st.write(f"--- Fetching project data for {project_id} ---")
        try:
            # 1. Fetch project info, including description
            project_response = self.supabase.table('projects').select("id, name, description").eq('id', project_id).single().execute()
            if not project_response.data:
                st.error(f"No project found with ID: {project_id}")
                st.stop()
            
            project_data = project_response.data
            ai_scores = {}
            agent_jobs_results = {}

            # 2. Fetch full analysis results from the 'agent_jobs' table
            st.write(f"--- Fetching analysis from agent_jobs for application {application_id} ---")
            
            job_types = ['founder', 'market']
            for job_type in job_types:
                job_res = self.supabase.table('agent_jobs').select("result").eq('application_id', application_id).eq('agent_type', job_type).eq('status', 'completed').order('created_at', desc=True).limit(1).single().execute()
                if job_res.data and job_res.data.get("result"):
                    agent_jobs_results[job_type] = job_res.data["result"]
                    if job_type == 'founder':
                        key_findings = job_res.data["result"].get("key_findings", {})
                        ai_scores["founder_score"] = key_findings.get("technical_expertise", 5) / 10.0
                    elif job_type == 'market':
                        ai_scores["market_score"] = job_res.data["result"].get("confidence", 0.5)

            final_scores = { "founder_score": 0.8, "market_score": 0.7, "technical_score": 0.75, "social_score": 0.9, "tokenomics_score": 0.6 }
            final_scores.update(ai_scores)
            project_data["ai_evaluation"] = final_scores
            
            return project_data, agent_jobs_results

        except Exception as e:
            st.error(f"Error fetching project data from Supabase: {e}")
            st.stop()

    def get_program(self, program_id: str) -> Dict[str, Any]:
        """Fetches program data from the 'funding_programs' table, including description."""
        st.write(f"--- Fetching program data for {program_id} ---")
        try:
            response = self.supabase.table('funding_programs').select("id, name, description, application_deadline_date").eq('id', program_id).single().execute()
            if response.data:
                program_data = response.data
                deadline_str = program_data.get("application_deadline_date")
                deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00")) if deadline_str else datetime.now(timezone.utc) + timedelta(days=30)
                return {
                    "id": program_data.get("id"),
                    "name": program_data.get("name", "Unknown Program"),
                    "description": program_data.get("description", "No description provided."),
                    "historical_acceptance_rate": program_data.get("historical_acceptance_rate", 0.05),
                    "deadline": deadline_dt,
                }
            else:
                st.error(f"No program found with ID: {program_id}")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching program data from Supabase: {e}")
            st.stop()

    def get_materials_for_application(self, application_id: str) -> List[Dict[str, Any]]:
        """Fetches all materials associated with a given application."""
        st.write(f"--- Fetching materials for application {application_id} ---")
        try:
            material_ids_response = self.supabase.table('application_materials').select('material_id').eq('application_id', application_id).execute()
            if not material_ids_response.data:
                st.warning(f"No materials found for application {application_id}")
                return []
            material_ids = [item['material_id'] for item in material_ids_response.data]
            materials_response = self.supabase.table('materials').select('name, content').in_('id', material_ids).execute()
            return materials_response.data if materials_response.data else []
        except Exception as e:
            st.error(f"Error fetching materials for application: {e}")
            return []


class ExternalAPIClient:
    """A client to interact with external APIs for real-time signals."""
    def __init__(self, github_token: str):
        self._github_token = github_token

    def get_current_signals(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Fetches real-time signals for a project."""
        # Mock data
        return {
            "github_commits_30d": 47,
            "social_sentiment_7d": 0.85,
            "token_performance_7d": None, # Assuming no token yet
        }


class FeatureAssembler:
    """Assembles features for the prediction model."""
    def assemble_features(
        self,
        project_data: Dict[str, Any],
        program_data: Dict[str, Any],
        realtime_signals: Dict[str, Any],
        user_weights: Dict[str, int],
    ) -> Dict[str, Any]:
        """Assembles a feature dictionary from various data sources."""
        supafund_scores = project_data["ai_evaluation"]
        
        # Normalize user weights to 0-1 scale for calculation
        normalized_weights = {k: v / 5.0 for k, v in user_weights.items()}
        
        weighted_score = sum(
            score * normalized_weights.get(dimension.replace('_score', ''), 0.0)
            for dimension, score in supafund_scores.items()
        )

        features = {
            "overall_weighted_score": weighted_score,
            "user_weight_config": user_weights,
            "program_fit_score": self._calculate_program_fit(project_data, program_data),
            "program_acceptance_rate": program_data["historical_acceptance_rate"],
            "application_completeness": self._score_application_quality(project_data),
            "days_before_deadline": (program_data["deadline"] - datetime.now(timezone.utc)).days,
            "github_activity_trend": realtime_signals["github_commits_30d"],
            "social_sentiment_recent": realtime_signals["social_sentiment_7d"],
            "market_momentum": realtime_signals["token_performance_7d"],
            **supafund_scores
        }
        if "raw_syn_data" in project_data:
            features["raw_syn_data"] = project_data["raw_syn_data"]
        return features

    def _calculate_program_fit(self, project_data: Dict, program_data: Dict) -> float:
        return 0.82

    def _score_application_quality(self, project_data: Dict) -> float:
        return 0.95


class PredictionEngine:
    """Generates predictions based on assembled features."""
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def predict(self, state: AgentState) -> tuple[Dict[str, Any], str]:
        """Generates a prediction using the LLM and returns the prediction data and the prompt used."""
        prompt = self._build_prediction_prompt(state)
        prediction_data = self._llm_client.make_prediction_request(prompt)
        
        features = state["features"]
        raw_ai_scores = {
            k: v for k, v in features.items() 
            if k.endswith('_score') and k not in ["overall_weighted_score", "program_fit_score"]
        }

        prediction_data["feature_breakdown"] = {
            "supafund_ai_scores": raw_ai_scores,
            "user_weighted_score": features["overall_weighted_score"],
            "program_fit_analysis": features["program_fit_score"],
            "timing_factors": features["days_before_deadline"],
            "realtime_signals": {
                "github_momentum": features["github_activity_trend"],
                "social_sentiment": features["social_sentiment_recent"]
            }
        }
        return prediction_data, prompt

    def _build_prediction_prompt(self, state: AgentState) -> str:
        """Builds a comprehensive prompt with all available context."""
        project_data = state["project_data"]
        program_data = state["program_data"]
        materials_data = state.get("materials_data", [])
        agent_jobs_results = state.get("agent_jobs_results", {})
        features = state["features"]
        market_question = state["market_question"]
        user_weights = state["user_weights"]

        # Format materials for the prompt
        formatted_materials = "\n\n".join(
            f"### Material: {material.get('name', 'Untitled')}\n\n{material.get('content', 'No content.')}"
            for material in materials_data
        )
        if not formatted_materials:
            formatted_materials = "No application materials submitted."

        # Format agent job results for the prompt
        formatted_agent_jobs = "\n\n".join(
            f"### {job_type.capitalize()} Analysis Report\n\n{result.get('raw_analysis', 'No detailed analysis available.')}"
            for job_type, result in agent_jobs_results.items()
        )
        if not formatted_agent_jobs:
            formatted_agent_jobs = "No internal AI analysis reports available."
            
        return f"""
Market Question: "{market_question}"

Based on the comprehensive information below, analyze the project's likelihood of being accepted into the funding program. Provide a JSON response.

--- PROGRAM DETAILS ---
Program Name: {program_data.get('name')}
Program Description:
{program_data.get('description')}

--- PROJECT DETAILS ---
Project Name: {project_data.get('name')}
Project Description:
{project_data.get('description')}

--- SUBMITTED MATERIALS ---
{formatted_materials}

--- INTERNAL AI ANALYSIS (Supafund Advantage) ---
{formatted_agent_jobs}

--- KEY METRICS & SCORES ---
Founder evaluation score: {features.get('founder_score', 0):.2f}
Market analysis score: {features.get('market_score', 0):.2f}
Technical assessment score: {features.get('technical_score', 0):.2f}
Program fit analysis score: {features.get('program_fit_score', 0):.2f}
Historical program acceptance rate: {features.get('program_acceptance_rate', 0):.1%}
Days until deadline: {features.get('days_before_deadline', 'N/A')}

--- USER EVALUATION PRIORITIES ---
You have been given specific evaluation priorities by the user. Here are their weights and what each level means:

{self._format_user_priorities(user_weights)}

--- FINAL INSTRUCTION ---
Analyze all the provided information. Prioritize the qualitative information from descriptions and analysis reports over just the scores. In your reasoning, explicitly reference how the project's materials and our internal analysis align (or misalign) with the program's description and goals.

Provide a JSON object with the following structure:
{{
    "prediction": "YES|NO",
    "confidence": float,
    "reasoning": "Detailed explanation incorporating all context, especially project/program fit."
}}
"""
    
    def _format_user_priorities(self, user_weights: Dict[str, int]) -> str:
        """Formats user priorities with detailed descriptions for the LLM prompt."""
        formatted_priorities = []
        
        for dimension, weight_value in user_weights.items():
            if dimension in WEIGHT_DEFINITIONS:
                config = WEIGHT_DEFINITIONS[dimension]
                level_info = config['levels'][weight_value]
                
                formatted_priorities.append(f"""
**{config['name']} (Priority Level: {weight_value}/5)**
- Focus Level: {level_info['label']}
- Evaluation Approach: {level_info['description']}
""")
        
        return "\n".join(formatted_priorities)

# --- LangGraph State and Workflow ---

@st.cache_resource
def get_pipeline():
    """Initializes and returns the SupafundDataPipeline."""
    llm_client = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))
    supafund_client = SupafundClient(url=SUPABASE_URL, key=SUPABASE_KEY)
    external_api_client = ExternalAPIClient(github_token=os.getenv("GITHUB_TOKEN"))
    feature_assembler = FeatureAssembler()
    prediction_engine = PredictionEngine(llm_client)

    return SupafundDataPipeline(
        supafund_client, external_api_client, feature_assembler, prediction_engine
    )

class SupafundDataPipeline:
    """Manages the end-to-end data processing and prediction workflow."""
    def __init__(
        self, supafund_client, external_api_client, feature_assembler, prediction_engine
    ):
        self._supafund_client = supafund_client
        self._external_api_client = external_api_client
        self._feature_assembler = feature_assembler
        self._prediction_engine = prediction_engine
        self.workflow = self._create_graph()

    # The methods below define the nodes of our graph
    def _get_application_data(self, state: AgentState) -> AgentState:
        """Node to fetch application, project, and program data."""
        try:
            application_id = state["application_id"]
            application = self._supafund_client.get_application(application_id)
            project, agent_jobs = self._supafund_client.get_project(application["project_id"], application["id"])
            program = self._supafund_client.get_program(application["program_id"])
            materials = self._supafund_client.get_materials_for_application(application["id"])
            
            state["project_data"] = project
            state["program_data"] = program
            state["agent_jobs_results"] = agent_jobs
            state["materials_data"] = materials
            state["error_message"] = None
        except Exception as e:
            state["error_message"] = f"Failed to get application data: {str(e)}"
        return state

    def _get_realtime_signals(self, state: AgentState) -> AgentState:
        project_data = state['project_data']
        signals = self._external_api_client.get_current_signals(project_data)
        return {**state, "realtime_signals": signals}

    def _assemble_features(self, state: AgentState) -> AgentState:
        with st.status("Assembling features...", expanded=True) as status:
            features = self._feature_assembler.assemble_features(
                project_data=state['project_data'],
                program_data=state['program_data'],
                realtime_signals=state['realtime_signals'],
                user_weights=state['user_weights']
            )
            status.update(label="Features assembled!", state="complete")
            return {**state, "features": features}

    def _make_prediction(self, state: AgentState) -> AgentState:
        """Node to generate a prediction using the LLM."""
        try:
            prediction_result, llm_prompt = self._prediction_engine.predict(state)
            state["prediction"] = prediction_result
            state["llm_prompt"] = llm_prompt
            state["error_message"] = None
        except Exception as e:
            state["error_message"] = f"Failed to make prediction: {str(e)}"
        return state
    
    def _create_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("get_application_data", self._get_application_data)
        graph.add_node("get_realtime_signals", self._get_realtime_signals)
        graph.add_node("assemble_features", self._assemble_features)
        graph.add_node("make_prediction", self._make_prediction)
        graph.set_entry_point("get_application_data")
        graph.add_edge("get_application_data", "get_realtime_signals")
        graph.add_edge("get_realtime_signals", "assemble_features")
        graph.add_edge("assemble_features", "make_prediction")
        graph.add_edge("make_prediction", END)
        return graph.compile()

# --- Streamlit UI ---

def main():
    """The main function to run the Streamlit application."""
    st.set_page_config(page_title="Supafund Prediction Pipeline", layout="wide")
    st.title("🤖 Supafund Prediction Pipeline Prototype")

    pipeline = get_pipeline()
    applications = pipeline._supafund_client.get_all_applications()

    # Create a mapping for the selectbox and ensure it's available
    app_options = {}
    if applications:
        app_options = {
            app['id']: f"{app.get('projects', {}).get('name', 'N/A')} - {app.get('funding_programs', {}).get('name', 'N/A')}"
            for app in applications
        }

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
        except ValueError:
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
                    inputs = {
                        "application_id": active_app_id,
                        "market_question": market_question,
                        "user_weights": user_weights,
                    }
                    final_state = pipeline.workflow.invoke(inputs)
                    # Store results in session state to display them
                    st.session_state.prediction_results = final_state
            elif not active_app_id:
                st.sidebar.error("Please select an application.")
            elif not weights_valid:
                st.sidebar.error("Please ensure total weights do not exceed 15 points.")
    else:
        st.sidebar.warning("No applications found to configure.")


    # --- Main Area ---
    st.header("All Applications")

    if not applications:
        st.info("No applications found in the database. When new applications are submitted, they will appear here.")
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