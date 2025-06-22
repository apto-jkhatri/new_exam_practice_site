import streamlit as st
import os
import json
import re
import random

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# --- Configuration ---
VECTOR_STORE_PATH = "faiss_dbt_docs_local_embeddings_index"
LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY_LENGTH = 10 # Max number of recent question texts to store
HISTORY_FOR_PROMPT = 3  # Max number of recent questions to include in the prompt

# --- Helper Functions ---

@st.cache_resource
def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        return None, None 
    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store, embeddings
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None, None

def get_relevant_documents(vector_store, query, k=5):
    if vector_store:
        try:
            results_with_scores = vector_store.similarity_search_with_score(query, k=k)
            # For debugging:
            # st.sidebar.subheader("Retrieved Documents:")
            # for i, (doc, score) in enumerate(results_with_scores):
            #     st.sidebar.caption(f"Doc {i+1} (Score: {score:.4f}) Source: {doc.metadata.get('source', 'N/A')}")
            #     st.sidebar.text_area(f"Content Snippet {i+1}", doc.page_content[:200]+"...", height=80, key=f"debug_doc_content_{i}_{hash(doc.page_content)}")
            return [doc for doc, score in results_with_scores]
        except Exception as e:
            st.error(f"Error during similarity search: {e}")
            return []
    return []

def generate_question_with_groq(client, context, topic, question_type, model_name="mixtral-8x7b-32768"):
    if not client:
        st.error("Groq client not initialized. Please ensure API key is set.")
        return None

    # Prepare history string for the prompt
    recent_questions_prompt_addition = ""
    if st.session_state.generated_question_history_texts:
        history_to_show = st.session_state.generated_question_history_texts[-HISTORY_FOR_PROMPT:]
        if history_to_show:
            recent_questions_prompt_addition = "\n\nTo ensure variety, please avoid generating questions that are too similar in theme or structure to the following recently generated examples:\n"
            for i, q_text in enumerate(history_to_show):
                recent_questions_prompt_addition += f"- \"{q_text}\"\n"

    common_instructions = f"""
You are an expert dbt exam question generator. Your goal is to create a new and unique challenging question
relevant to the dbt Analytics Engineering Certification Exam, based on the provided dbt documentation context
and the given topic: "{topic}".
{recent_questions_prompt_addition}
The question should accurately reflect dbt concepts and best practices.
Output the result in a VALID JSON format ONLY. Do not add any text before or after the JSON block.
Ensure all specified fields for the question type are present in the JSON.
"""

    if question_type == "multiple_choice":
        system_prompt = f"""
{common_instructions}
Question Type: Standard Multiple Choice.
The question should have 4 options (A, B, C, D), with only one correct answer.
For each option (correct or incorrect), you MUST provide a detailed explanation, referencing dbt principles or the provided context.

JSON Output Format:
{{
  "question_type": "multiple_choice",
  "scenario": "Optional: A brief scenario if applicable, otherwise null or omit.",
  "question": "The main question text?",
  "options": {{ "A": "Text A", "B": "Text B", "C": "Text C", "D": "Text D" }},
  "correct_answer_key": "C",
  "explanation_A": "Why A is correct/incorrect.",
  "explanation_B": "Why B is correct/incorrect.",
  "explanation_C": "Why C is correct/incorrect.",
  "explanation_D": "Why D is correct/incorrect."
}}
"""
    elif question_type == "scenario_based_multiple_choice":
        system_prompt = f"""
{common_instructions}
Question Type: Scenario-Based Multiple Choice.
First, describe a specific dbt scenario, problem, or provide a small YAML/SQL code snippet relevant to the topic and context.
Then, pose a multiple-choice question related to that scenario/code, with 4 options (A, B, C, D).
For each option, provide a detailed explanation.

JSON Output Format:
{{
  "question_type": "scenario_based_multiple_choice",
  "scenario": "Detailed scenario description. If including code/YAML, format it clearly using markdown backticks for code blocks if appropriate within the JSON string, or describe it textually.",
  "question": "Question related to the scenario?",
  "options": {{ "A": "Text A", "B": "Text B", "C": "Text C", "D": "Text D" }},
  "correct_answer_key": "C",
  "explanation_A": "Why A is correct/incorrect based on the scenario.",
  "explanation_B": "Why B is correct/incorrect based on the scenario.",
  "explanation_C": "Why C is correct/incorrect based on the scenario.",
  "explanation_D": "Why D is correct/incorrect based on the scenario."
}}
"""
    elif question_type == "fill_in_the_blank":
        system_prompt = f"""
{common_instructions}
Question Type: Fill-in-the-Blank.
Create a question with one or two key dbt terms or concepts missing, indicated by `____BLANK____`.
Provide the correct answer(s) for the blank(s). If multiple blanks, list answers in order.
Provide a brief explanation for the answer.

JSON Output Format:
{{
  "question_type": "fill_in_the_blank",
  "question": "A dbt ____BLANK____ is defined in a .sql file in the models directory, while a dbt ____BLANK____ is defined in a .csv file in the seeds directory.",
  "answers": ["model", "seed"],
  "explanation": "Models contain SQL transformations. Seeds are CSV files that dbt can load as tables, often for reference data."
}}
"""
    elif question_type == "arranging_steps":
        system_prompt = f"""
{common_instructions}
Question Type: Arranging Steps.
Describe a common dbt task or workflow scenario.
Provide 4-5 jumbled steps (each with a unique key like S1, S2) to complete the task.
The user will need to arrange these step keys in the correct order.
Provide the correct order of these keys and an explanation for this order.

JSON Output Format:
{{
  "question_type": "arranging_steps",
  "scenario": "To add a new not-null test to an existing column 'user_id' in the model 'dim_customers', what is the correct sequence of actions?",
  "jumbled_steps": {{
    "S1": "Run `dbt test` to verify the new test passes.",
    "S2": "Add the not-null test configuration under the 'user_id' column in the relevant schema.yml file for 'dim_customers'.",
    "S3": "Commit the changes to your git repository.",
    "S4": "Open the schema.yml file associated with the 'dim_customers' model (or create it if it doesn't exist)."
  }},
  "correct_order_keys": ["S4", "S2", "S1", "S3"],
  "explanation": "First open/create the YML (S4), then add the test config (S2), then run the test to verify (S1), and finally commit (S3)."
}}
"""
    else:
        st.error(f"Unknown question type: {question_type}")
        return None

    user_prompt = f"""
Here is the relevant dbt documentation context:
--- CONTEXT START ---
{context}
--- CONTEXT END ---

Generate one question of type "{question_type}" based ONLY on the provided context and the specified topic: "{topic}".
Ensure the generated question content is directly supported by the provided context.
Remember the JSON output format and provide all required fields for this question type.
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model_name,
            temperature=0.7, # Slightly higher temp for variety
            top_p=0.9,       # Using top_p can also help
        )
        response_content = chat_completion.choices[0].message.content
        
        if response_content.strip().startswith("```json"):
            response_content = response_content.strip()[7:]
            if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3]
        
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            json_response_str = json_match.group(0)
        else:
            json_response_str = response_content.strip()

        try:
            question_data = json.loads(json_response_str)
            if "question_type" not in question_data or question_data["question_type"] != question_type:
                 st.error(f"LLM response type mismatch or missing. Expected {question_type}, got {question_data.get('question_type')}. Raw Content: {json_response_str[:500]}")
                 return None
            
            # Add new question text to history (if valid question_data and question text exists)
            new_question_text = question_data.get("question")
            if new_question_text:
                st.session_state.generated_question_history_texts.append(new_question_text)
                if len(st.session_state.generated_question_history_texts) > MAX_HISTORY_LENGTH:
                    st.session_state.generated_question_history_texts = st.session_state.generated_question_history_texts[-MAX_HISTORY_LENGTH:]
            
            return question_data
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response from LLM: {e}. Raw response snippet: {json_response_str[:500]}")
            return None
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="dbt Exam Practice")
st.title("📚 dbt Exam Practice Platform")
st.markdown("Practice for your dbt exam with AI-generated questions based on the official documentation.")

# --- Session State Initialization ---
if 'api_key_confirmed' not in st.session_state: st.session_state.api_key_confirmed = False
if 'user_api_key' not in st.session_state: st.session_state.user_api_key = ""
if 'groq_client' not in st.session_state: st.session_state.groq_client = None
if 'question_data' not in st.session_state: st.session_state.question_data = None
if 'user_answer' not in st.session_state: st.session_state.user_answer = None
if 'user_ordered_steps' not in st.session_state: st.session_state.user_ordered_steps = []
if 'show_feedback' not in st.session_state: st.session_state.show_feedback = False
if 'generated_question_history_texts' not in st.session_state: st.session_state.generated_question_history_texts = []


# --- API Key Management (User Input Focused) ---
with st.expander("🔑 Groq API Key Configuration", expanded=not st.session_state.api_key_confirmed):
    if st.session_state.api_key_confirmed:
        st.success("Groq API Key is active.")
        if st.button("Change API Key", key="change_api_key_btn"):
            st.session_state.api_key_confirmed = False; st.session_state.user_api_key = ""; st.session_state.groq_client = None
            st.rerun()
    else:
        st.markdown("This app uses Groq to generate questions. Please provide your Groq API key.")
        user_api_key_input = st.text_input(
            "Enter your Groq API Key here:", type="password", value=st.session_state.user_api_key,
            key="main_api_key_input_field", help="Your API key is processed in this browser session and not stored by this application."
        )
        if st.button("Set and Verify API Key", key="set_api_key_btn"):
            if user_api_key_input:
                try:
                    client_test = Groq(api_key=user_api_key_input); client_test.models.list() 
                    st.session_state.groq_client = client_test; st.session_state.user_api_key = user_api_key_input
                    st.session_state.api_key_confirmed = True; st.success("Groq API key set and verified!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid or non-functional Groq API Key: {e}"); st.session_state.api_key_confirmed = False; st.session_state.groq_client = None
            else: st.warning("Please enter an API key.")

# --- Load Vector Store ---
vector_store = None
if st.session_state.api_key_confirmed:
    vs_load_result, _ = load_vector_store() 
    if vs_load_result: vector_store = vs_load_result

if not st.session_state.api_key_confirmed:
    st.info("Please configure your Groq API Key above to enable question generation.")
elif not vector_store and st.session_state.api_key_confirmed:
    st.warning("Vector store could not be loaded. Ensure `faiss_dbt_docs_local_embeddings_index` is present in the app directory.")

# --- Main Application Area (Conditionally Displayed) ---
if st.session_state.api_key_confirmed and vector_store:
    st.sidebar.title("⚙️ Exam Settings")
    dbt_topics_structured = {
        "Topic 1: Developing dbt models": ["Identifying and verifying raw object dependencies", "Understanding core dbt materializations", "Conceptualizing modularity and DRY principles", "Converting business logic into performant SQL", "Using commands: run, test, docs, seed", "Creating logical model flow and clean DAGs", "Defining configurations in dbt_project.yml", "Configuring sources in dbt", "Using dbt Packages", "Utilizing git functionality in development", "Creating Python models in dbt", "Providing access with 'grants' configuration"],
        "Topic 2: Understanding dbt models governance": ["Adding contracts to models", "Creating and deprecating model versions", "Configuring model access"],
        "Topic 3: Debugging data modeling errors": ["Understanding logged error messages", "Troubleshooting with compiled code", "Troubleshooting .yml compilation errors", "Distinguishing pure SQL vs. dbt issues", "Developing, implementing, and testing fixes"],
        "Topic 4: Managing data pipelines": ["Troubleshooting DAG failure points", "Using dbt clone", "Troubleshooting errors from integrated tools"],
        "Topic 5: Implementing dbt tests": ["Using generic, singular, custom, and custom generic tests", "Testing assumptions for models and sources", "Implementing testing steps in workflow"],
        "Topic 6: Creating and Maintaining dbt documentation": ["Updating dbt docs", "Implementing source, table, column descriptions in .yml", "Using macros for lineage on the DAG"],
        "Topic 7: Implementing and maintaining external dependencies": ["Implementing dbt exposures", "Implementing source freshness"],
        "Topic 8: Leveraging the dbt state": ["Understanding dbt state", "Using dbt retry", "Combining state and result selectors"]
    }
    dbt_topics_for_selectbox = []; 
    for main_topic, sub_topics in dbt_topics_structured.items():
        dbt_topics_for_selectbox.append(main_topic)
        for sub_topic in sub_topics: dbt_topics_for_selectbox.append(f"  - {sub_topic} ({main_topic.split(':')[0]})")
    selected_topic_str = st.sidebar.selectbox("Select a dbt Topic:", dbt_topics_for_selectbox, index=0, key="topic_select_detailed")
    custom_topic = st.sidebar.text_input("Or enter a custom topic/keyword:", placeholder="e.g., ephemeral models", key="custom_topic_input")
    final_topic_query = custom_topic if custom_topic else selected_topic_str
    question_types = ["multiple_choice", "scenario_based_multiple_choice", "fill_in_the_blank", "arranging_steps"]
    selected_question_type = st.sidebar.selectbox("Select Question Type:", question_types, index=0, key="q_type_select_detailed")
    available_models = ["mistral-saba-24b", "llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it"]
    selected_model = st.sidebar.selectbox("Select LLM Model (Groq):", available_models, index=0, key="model_select_groq")

    if st.button("🔄 Generate New Question", type="primary", use_container_width=False, key="generate_q_btn"):
        current_groq_client = st.session_state.groq_client
        if not current_groq_client: st.error("Groq client not available. Please re-verify API key.")
        else:
            with st.spinner(f"Generating '{selected_question_type}' question on '{final_topic_query}' using {selected_model}..."):
                # Vary context slightly for more variety
                k_chunks = random.choice([4, 5, 6]) 
                all_retrieved_docs = get_relevant_documents(vector_store, final_topic_query, k=k_chunks + 2) # fetch a bit more
                if len(all_retrieved_docs) >= k_chunks:
                    retrieved_docs = random.sample(all_retrieved_docs, k_chunks)
                elif all_retrieved_docs: 
                    retrieved_docs = all_retrieved_docs
                else: retrieved_docs = []
                
                if retrieved_docs:
                    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    source_urls = list(set([doc.metadata.get("source", "N/A") for doc in retrieved_docs if doc.metadata.get("source")]))
                    primary_source_url = source_urls[0] if source_urls else "Documentation source not found"
                    
                    generated_data = generate_question_with_groq(current_groq_client, context_str, final_topic_query, selected_question_type, model_name=selected_model)
                    
                    if generated_data:
                        generated_data["source_url_for_explanation"] = primary_source_url
                        generated_data["all_context_source_urls"] = source_urls
                        st.session_state.question_data = generated_data
                    else: st.session_state.question_data = None
                    st.session_state.user_answer = None; st.session_state.user_ordered_steps = []; st.session_state.show_feedback = False
                else: st.error("Could not retrieve relevant context. Try a different topic or check vector store.")
            st.rerun()

    if st.session_state.question_data:
        q_data = st.session_state.question_data; q_type = q_data.get("question_type")
        st.subheader(f"Question: ({q_type.replace('_', ' ').title()})")
        question_hash = hash(str(q_data.get("question", ""))+str(q_data.get("scenario", "")))

        def display_source_links(data):
            source_url_display = data.get("source_url_for_explanation", "No specific source URL found.")
            all_urls = data.get("all_context_source_urls", [])
            if source_url_display != "No specific source URL found." and source_url_display != "N/A":
                st.markdown(f"**Relevant Documentation:** [{source_url_display}]({source_url_display})")
            elif all_urls:
                st.markdown("**Context Source(s) (pages used as context for the question):**")
                for url_item in [url for url in all_urls if url != "N/A"]: st.markdown(f"- [{url_item}]({url_item})")
        
        if q_type == "multiple_choice" or q_type == "scenario_based_multiple_choice":
            if q_type == "scenario_based_multiple_choice" and "scenario" in q_data: st.markdown(f"**Scenario:** {q_data['scenario']}")
            st.markdown(f"**{q_data['question']}**"); options = q_data['options']; option_keys = list(options.keys())
            radio_key = f"user_choice_radio_{question_hash}"
            if not st.session_state.show_feedback:
                st.session_state.user_answer = st.radio("Choose your answer:", options=option_keys, format_func=lambda key: f"{key}: {options[key]}", index=None, key=radio_key)
                if st.button("✅ Submit Answer", use_container_width=True, key=f"submit_{radio_key}"):
                    if st.session_state.user_answer is None: st.warning("Please select an answer.")
                    else: st.session_state.show_feedback = True; st.rerun()
            if st.session_state.show_feedback and st.session_state.user_answer is not None:
                is_correct = st.session_state.user_answer == q_data['correct_answer_key']
                st.markdown(f"---"); st.markdown(f"**Your Answer:** {st.session_state.user_answer}: {options[st.session_state.user_answer]}")
                if is_correct: st.success("🎉 Correct!")
                else: st.error(f"❌ Incorrect. Correct answer was {q_data['correct_answer_key']}.")
                display_source_links(q_data)
                st.markdown(f"---"); st.subheader("Explanations:")
                for key_opt in option_keys:
                    exp_key = f"explanation_{key_opt}"; exp_text = q_data.get(exp_key, "N/A"); prefix = ""
                    if key_opt == st.session_state.user_answer: prefix += "➡️ **Your Choice:** "
                    if key_opt == q_data['correct_answer_key']: prefix += "✅ **Correct Answer:** "
                    with st.expander(f"{prefix}Option {key_opt}: {options[key_opt]}", expanded=(key_opt == st.session_state.user_answer or key_opt == q_data['correct_answer_key'])): st.markdown(exp_text)

        elif q_type == "fill_in_the_blank":
            st.markdown(q_data['question'].replace("____BLANK____", " **\[ ______ \]** "))
            text_input_key = f"fill_blank_input_{question_hash}"
            if not st.session_state.show_feedback:
                st.session_state.user_answer = st.text_input("Your answer(s) (comma-separated if multiple):", key=text_input_key)
                if st.button("✅ Submit Answer", use_container_width=True, key=f"submit_{text_input_key}"):
                    if not st.session_state.user_answer: st.warning("Please enter your answer.")
                    else: st.session_state.show_feedback = True; st.rerun()
            if st.session_state.show_feedback and st.session_state.user_answer is not None:
                user_answers_list = [ans.strip().lower() for ans in st.session_state.user_answer.split(',')]; correct_answers_list = [str(ans).strip().lower() for ans in q_data['answers']]; is_correct = user_answers_list == correct_answers_list
                st.markdown(f"---"); st.markdown(f"**Your Answer(s):** {', '.join(user_answers_list)}"); st.markdown(f"**Correct Answer(s):** {', '.join(correct_answers_list)}")
                if is_correct: st.success("🎉 Correct!")
                else: st.error(f"❌ Incorrect.")
                st.subheader("Explanation:"); st.markdown(q_data['explanation'])
                display_source_links(q_data)

        elif q_type == "arranging_steps":
            st.markdown(f"**Scenario:** {q_data['scenario']}"); st.markdown("**Arrange the following steps in the correct order:**"); jumbled_steps_dict = q_data['jumbled_steps']
            for key_step, step_text in jumbled_steps_dict.items(): st.markdown(f"- **{key_step}:** {step_text}")
            arrange_input_key = f"arrange_steps_input_{question_hash}"
            if not st.session_state.show_feedback:
                st.session_state.user_answer = st.text_input(f"Enter step keys in order, comma-separated (e.g., {','.join(jumbled_steps_dict.keys())}):", key=arrange_input_key)
                if st.button("✅ Submit Answer", use_container_width=True, key=f"submit_{arrange_input_key}"):
                    if not st.session_state.user_answer: st.warning("Please enter the order of steps.")
                    else: st.session_state.show_feedback = True; st.rerun()
            if st.session_state.show_feedback and st.session_state.user_answer is not None:
                user_ordered_keys = [key.strip().upper() for key in st.session_state.user_answer.split(',')]; correct_order_keys = q_data['correct_order_keys']; is_correct = user_ordered_keys == correct_order_keys
                st.markdown(f"---"); st.markdown(f"**Your Order:** {', '.join(user_ordered_keys)}"); st.markdown(f"**Correct Order:** {', '.join(correct_order_keys)}")
                if is_correct: st.success("🎉 Correct!")
                else: st.error(f"❌ Incorrect.")
                st.subheader("Explanation of Correct Order:"); st.markdown(q_data['explanation'])
                display_source_links(q_data)

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.caption("dbt Exam Practice App - Use at your own discretion. LLM-generated content may not always be perfect.")