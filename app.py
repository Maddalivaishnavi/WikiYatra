import streamlit as st
import requests
from PIL import Image
import io
import json
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import time

# Load environment variables
load_dotenv()

# Constants
WIKIMEDIA_HEADERS = {'User-Agent': 'WikiYatra/1.0 (vaishnavi.maddali.project@example.com)'}
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_ID = "gemma3"

# --- Helper: Ollama Query ---
def query_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL_ID, "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"❌ Ollama Error: {e}"

# --- Helper: Wikipedia Fetch ---
def fetch_wikipedia_content(title, lang="en"):
    try:
        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(summary_url, headers=WIKIMEDIA_HEADERS, timeout=15)
        if response.status_code == 200:
            data = response.json()
            summary = data.get("extract", "")
            title = data.get("title", "")
            description = data.get("description", "")
            return summary, title, description
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Wikipedia fetch error: {e}")
        return None, None, None

# --- Tabs ---
tab1, tab2, tab_chat = st.tabs(["🌍 Destination Search", "🗺️ Route Planner", "💬 AI Chatbot"])

# -------- Destination Search Tab --------
with tab1:
    st.header("🌍 Explore Indian Destinations")

    destination = st.text_input("Enter a destination in India (e.g., Goa, Jaipur, Hampi)", "")
    lang = st.text_input("Choose Language", value="en")

    if destination:
        with st.spinner("🔎 Searching Wikipedia..."):
            summary, image_url, wiki_title = fetch_wikipedia_content(destination, lang)

            # Fallback to English if no data found
            if not summary:
                st.warning(f"No data found for '{destination}' in '{lang}'. Trying English...")
                summary, image_url, wiki_title = fetch_wikipedia_content(destination, "en")

            # If still no summary found
            if not summary:
                st.error("❌ No data found. Please check the spelling or try another place.")
            else:
                st.subheader(f"📌 {wiki_title}")
                st.markdown(summary)

                if image_url:
                    st.image(image_url, caption=wiki_title)

# --- Tab 2: Route Planner (using Leaflet) ---
with tab2:
    st.header("🗺️ Plan Your Route in India")
    origin = st.text_input("Enter starting location")
    destination = st.text_input("Enter destination")

    if st.button("Show Route"):
        if origin and destination:
            # Dummy coordinates for demo
            origin_coords = [20.5937, 78.9629]  # India center
            dest_coords = [21.1458, 79.0882]    # Nagpur

            m = folium.Map(location=origin_coords, zoom_start=5)
            folium.Marker(origin_coords, tooltip="Start", popup=origin, icon=folium.Icon(color='green')).add_to(m)
            folium.Marker(dest_coords, tooltip="End", popup=destination, icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine([origin_coords, dest_coords], color="blue", weight=2.5).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.warning("Please enter both locations")

# --- Tab 3: AI Chatbot ---
with tab_chat:
    st.header("💬 AI Chatbot (Powered by Ollama + Wikipedia)")
    st.write("Ask about any Indian place (e.g., 'Tell me about Hampi', 'Best places in Kerala')")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about Indian destinations...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    identify_prompt = (
                        f"Identify the Indian tourist destination in the question below. "
                        f"Return just the name of the place. If no valid place is found, return 'NONE'.\n\n"
                        f"Question: {user_input}"
                    )
                    place = query_ollama(identify_prompt).strip()

                    if place.lower() == "none" or not place:
                        reply = "I'm sorry, I couldn't identify a specific Indian destination in your question."
                    else:
                        wiki_summary, wiki_title, wiki_desc = fetch_wikipedia_content(place, lang="en")

                        if not wiki_summary:
                            reply = f"❌ I couldn't find info for **{place}**. Try another."
                        else:
                            answer_prompt = (
                                f"The user asked: '{user_input}'.\n\n"
                                f"Here is some background info from Wikipedia about {place}:\n\n"
                                f"{wiki_summary}\n\n"
                                f"Based on this, answer the user's question clearly and helpfully."
                            )
                            reply = query_ollama(answer_prompt).strip()
                except Exception as e:
                    reply = f"❌ Ollama Error: {e}"

                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
