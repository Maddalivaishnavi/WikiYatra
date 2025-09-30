import streamlit as st
import requests
from PIL import Image
import io
import json
import os
import re
from dotenv import load_dotenv
import googlemaps
import folium
from streamlit_folium import st_folium
import polyline
import time

# Load environment variables
load_dotenv()

# --- Configuration ---
WIKIPEDIA_API_BASE_URL_PREFIX = ".wikipedia.org/w/api.php"
ENGLISH_SUMMARY_SENTENCES = 5

WIKIMEDIA_HEADERS = {
    'User-Agent': 'WikiYatra/1.0 (vaishnavi.maddali.project@example.com)'
}

# --- Mapping & Services Configuration ---
Maps_API_KEY = os.getenv("Maps_API_KEY")
gmaps = googlemaps.Client(key=Maps_API_KEY) if Maps_API_KEY else None

# --- LLM Chatbot Configuration ---
HF_API_KEY = os.getenv("HF_API_KEY")
LLM_MODEL_TYPE = None
HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/"

# List of available models in order of preference
AVAILABLE_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/flan-t5-large",
    "facebook/blenderbot-400M-distill",
    "gpt2"
]

if HF_API_KEY:
    LLM_MODEL_TYPE = "huggingface_inference"

REQUEST_TIMEOUT = 30
API_REQUEST_DELAY = 1.5

# Basic location information for fallback
BASIC_LOCATION_INFO = {
    "hyderabad": "Hyderabad is the capital of Telangana state in southern India. It's known for its historic landmarks like Charminar and Golconda Fort, as well as being a major center for IT and business.",
    "delhi": "Delhi is the capital of India, a bustling metropolis with historic sites like Red Fort and Qutub Minar alongside modern government buildings.",
    "mumbai": "Mumbai is India's financial capital, home to Bollywood and landmarks like the Gateway of India.",
    "bangalore": "Bangalore (officially Bengaluru) is the capital of Karnataka and India's leading technology hub.",
    "chennai": "Chennai is the capital of Tamil Nadu, known for its beaches, temples, and cultural heritage.",
    "kolkata": "Kolkata is the capital of West Bengal, known for its colonial architecture and cultural scene.",
    "jaipur": "Jaipur is the capital of Rajasthan, known as the 'Pink City' for its historic buildings.",
    "ahmedabad": "Ahmedabad is the largest city in Gujarat, known for its textile industry and Gandhi heritage.",
    "pune": "Pune is a major city in Maharashtra, known for its educational institutions and IT industry.",
    "hampi": "Hampi is a UNESCO World Heritage Site in Karnataka with ancient ruins from the Vijayanagara Empire."
}

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_wikipedia_content(query, lang="en", is_fallback_attempt=False):
    """Fetch Wikipedia content with fallback logic"""
    api_url = f"https://{lang}{WIKIPEDIA_API_BASE_URL_PREFIX}"

    extract_params = {
        "explaintext": True,
        "redirects": 1,
    }
    if lang == "en":
        extract_params["exsentences"] = ENGLISH_SUMMARY_SENTENCES
    else:
        extract_params["exlimit"] = 1

    params_title = {
        "action": "query",
        "format": "json",
        "prop": "extracts|pageimages|info",
        "pithumbsize": 300,
        "inprop": "url",
        "titles": query,
        "uselang": lang,
        **extract_params
    }
    
    try:
        response = requests.get(api_url, params=params_title, headers=WIKIMEDIA_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        if pages and list(pages.keys())[0] != '-1':
            page_id = next(iter(pages))
            page = pages[page_id]
            summary = page.get("extract")
            image_url = page.get("thumbnail", {}).get("source")
            full_url = page.get("fullurl")

            if summary:
                if lang != "en" and LLM_MODEL_TYPE == "huggingface_inference":
                    processed_summary = ai_summarize(summary, target_lang=lang)
                else:
                    processed_summary = summary
                return processed_summary, image_url, full_url
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error during direct title lookup for '{query}' in {lang.upper()}: {e}")
    except Exception as e:
        st.warning(f"Error during direct title lookup for '{query}' in {lang.upper()}: {e}")

    # Attempt search fallback
    if lang != "en" or is_fallback_attempt:
        params_search = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "srprop": "snippet|titlesnippet",
            "uselang": lang
        }
        try:
            response_search = requests.get(api_url, params=params_search, headers=WIKIMEDIA_HEADERS, timeout=REQUEST_TIMEOUT)
            response_search.raise_for_status()
            data_search = response_search.json()
            
            search_results = data_search.get("query", {}).get("search", [])

            if search_results:
                found_title = search_results[0].get("title")
                if found_title:
                    st.info(f"Found '{found_title}' via search in {lang.upper()}. Fetching details...")
                    return fetch_wikipedia_content(found_title, lang=lang, is_fallback_attempt=True)

        except requests.exceptions.RequestException as e:
            st.warning(f"Network error during search lookup for '{query}' in {lang.upper()}: {e}")
        except Exception as e:
            st.warning(f"Error during search lookup for '{query}' in {lang.upper()}: {e}")

    # Fallback to English Wikipedia
    if lang != "en" and not is_fallback_attempt:
        st.warning(f"No direct or search results found for '{query}' in {lang.upper()}. Attempting to fetch from English Wikipedia.")
        return fetch_wikipedia_content(query, "en", is_fallback_attempt=True)

    return None, None, None

def query_huggingface_api(payload, model_id=None):
    """Improved API query with automatic fallback and retry"""
    if not HF_API_KEY:
        st.error("Hugging Face API Key is not set. Cannot query model.")
        return None
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Try models in order of preference
    for model in [model_id] if model_id else AVAILABLE_MODELS:
        try:
            response = requests.post(
                f"{HF_INFERENCE_API_URL}{model}",
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                continue  # Silently try next model
            elif "loading" in response.text.lower():
                # Wait longer and retry the same model
                time.sleep(15)
                response = requests.post(
                    f"{HF_INFERENCE_API_URL}{model}",
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                if response.status_code == 200:
                    return response.json()
                continue
            elif response.status_code == 503:
                # Service unavailable - wait and try next model
                time.sleep(5)
                continue
                
        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue
    
    st.error("All models are currently busy. Please try again in a moment.")
    return None

def ai_summarize(text, target_lang="en"):
    """Simplified summarization that works with all models"""
    if not text or not HF_API_KEY:
        return text or "No content available"
    
    prompt_text = f"Summarize this in {target_lang} (1-2 paragraphs):\n{text}\nSummary:"
    
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    response = query_huggingface_api(payload)
    return response[0]['generated_text'] if response else text

def get_coordinates(location_name):
    """Fetch coordinates for a location"""
    location_name = location_name.strip()
    if not location_name:
        st.error("Location name cannot be empty.")
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{location_name}, India",
        'format': 'json',
        'limit': 1,
        'countrycodes': 'in',
        'addressdetails': 1
    }
    try:
        time.sleep(API_REQUEST_DELAY)
        response = requests.get(url, params=params, headers=WIKIMEDIA_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            st.warning(f"No results found for '{location_name}'. Please try a more specific name.")
            return None
            
    except requests.exceptions.Timeout:
        st.error(f"Request timed out for '{location_name}'. Please try again.")
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            st.error(f"Rate limit exceeded for geocoding '{location_name}'. Please wait a moment and try again.")
        else:
            st.error(f"HTTP error occurred during geocoding for '{location_name}': {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Network error occurred during geocoding for '{location_name}': {req_err}")
    except json.JSONDecodeError:
        st.error(f"Invalid response format from geocoding service for '{location_name}'.")
    except Exception as e:
        st.error(f"Unexpected error geocoding '{location_name}': {e}")
    return None

def get_route(start_coords, end_coords):
    """Fetch driving route between coordinates"""
    if not start_coords or not end_coords:
        return None, None, None
    
    url = f"https://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    params = {
        'overview': 'full',
        'geometries': 'polyline',
        'steps': 'false'
    }
    try:
        time.sleep(API_REQUEST_DELAY)
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        if 'routes' in data and data['routes']:
            route = data['routes'][0]
            distance_km = route['distance'] / 1000
            duration_hours = route['duration'] / 3600
            return distance_km, duration_hours, route.get('geometry')
        else:
            st.warning("No route found between the specified locations.")
            return None, None, None
            
    except requests.exceptions.Timeout:
        st.error("Route request timed out. Please try again.")
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            st.error("Rate limit exceeded for routing. Please wait a moment and try again.")
        else:
            st.error(f"HTTP error occurred fetching route: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Network error occurred fetching route: {req_err}")
    except json.JSONDecodeError:
        st.error("Invalid response format from routing service.")
    except Exception as e:
        st.error(f"Unexpected error fetching route: {e}")
    return None, None, None

def get_places_nearby(lat, lon, place_type="food", radius=5000):
    """
    Improved function to fetch nearby places with better error handling and more complete results
    """
    # Expanded mapping of place types to OSM tags
    osm_amenity_map = {
        "food": ["amenity=restaurant", "amenity=cafe", "amenity=fast_food", "amenity=bar"],
        "hotels": ["tourism=hotel", "tourism=guest_house", "tourism=hostel", "tourism=motel"],
        "fuel": ["amenity=fuel", "amenity=charging_station"]
    }
    
    # Build the overpass query
    overpass_url = "https://overpass-api.de/api/interpreter"
    queries = []
    
    for tag in osm_amenity_map.get(place_type, ["amenity=restaurant"]):
        queries.append(f'node[{tag}](around:{radius},{lat},{lon});')
        queries.append(f'way[{tag}](around:{radius},{lat},{lon});')
        queries.append(f'relation[{tag}](around:{radius},{lat},{lon});')
    
    overpass_query = f"""
    [out:json][timeout:30];
    (
        {"".join(queries)}
    );
    out center;
    """
    
    try:
        time.sleep(API_REQUEST_DELAY)
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        places = []
        
        if data.get("elements"):
            for el in data["elements"]:
                name = el.get("tags", {}).get("name", "Unnamed Place")
                amenity = el.get("tags", {}).get("amenity", "")
                tourism = el.get("tags", {}).get("tourism", "")
                
                # Get more complete address information
                address_parts = []
                if "addr:street" in el.get("tags", {}):
                    address_parts.append(el["tags"]["addr:street"])
                if "addr:housenumber" in el.get("tags", {}):
                    address_parts.append(el["tags"]["addr:housenumber"])
                
                address = ", ".join(address_parts) if address_parts else ""
                
                # Create a more informative display string
                display_str = name
                if address:
                    display_str += f" ({address})"
                elif amenity:
                    display_str += f" ({amenity.replace('_', ' ').title()})"
                elif tourism:
                    display_str += f" ({tourism.replace('_', ' ').title()})"
                
                places.append(display_str)
        
        return places[:15]  # Return more results and let the UI limit them
        
    except requests.exceptions.Timeout:
        st.error(f"Request timed out while searching for {place_type}. Please try again.")
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            st.error(f"Rate limit exceeded while searching for {place_type}. Please wait and try again.")
        else:
            st.error(f"HTTP error occurred finding {place_type}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Network error occurred finding {place_type}: {req_err}")
    except json.JSONDecodeError:
        st.error(f"Invalid response format while searching for {place_type}.")
    except Exception as e:
        st.error(f"Unexpected error finding {place_type}: {e}")
    
    return []

def format_duration(hours):
    """Format duration in hours to human-readable format"""
    if hours <= 0:
        return "Less than a minute"
    
    total_minutes = int(hours * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    
    if hh > 0:
        return f"{hh}h {mm}m" if mm > 0 else f"{hh}h"
    else:
        return f"{mm}m"

def is_route_query(prompt: str) -> bool:
    """Check if the prompt is asking for a route between locations"""
    route_keywords = ["route", "how to reach", "how to go", "from.*to", "distance", "travel", "get to", "way to"]
    return any(re.search(keyword, prompt.lower()) for keyword in route_keywords)

def extract_route_locations(prompt: str) -> tuple:
    """Extract start and end locations from route queries"""
    prompt = prompt.lower()
    if "from" in prompt and "to" in prompt:
        try:
            start = prompt.split("from")[1].split("to")[0].strip()
            end = prompt.split("to")[1].strip()
            # Remove any question marks or extra words
            end = re.sub(r'[?].*', '', end).strip()
            return (start, end)
        except:
            return (None, None)
    return (None, None)

def format_wikipedia_response(place: str, summary: str, url: str) -> str:
    """Format Wikipedia response nicely"""
    response = f"Here's what I found about {place}:\n\n{summary[:500]}"
    if len(summary) > 500:
        response += "..."
    if url:
        response += f"\n\n[Read more on Wikipedia]({url})"
    return response

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.set_page_config(
    page_title="WikiYatra - Smart Cultural Travel Assistant",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state
if 'route_calculated' not in st.session_state:
    st.session_state.route_calculated = False
if 'route_data' not in st.session_state:
    st.session_state.route_data = None

# Sidebar - Settings
with st.sidebar:
    st.header("Settings")
    selected_language = st.selectbox(
        "Select Content Language:",
        options=["English", "Hindi", "Telugu", "Tamil", "Kannada", 
                "Malayalam", "Gujarati", "Bengali", "Marathi", "Punjabi"],
        index=0,
        key="lang_select"
    )

# Language codes mapping
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Marathi": "mr",
    "Punjabi": "pa"
}

# Main content
st.title("🧭 WikiYatra: Your Smart Cultural Travel Assistant")
st.markdown("Explore India's culturally rich destinations with ease and depth.")

# Tabs
tab_info, tab_route, tab_chat = st.tabs(["🏛️ Destination Info", "🗺️ Route Planner", "💬 AI Chatbot"])

# ----------- Destination Info Tab (Horizontal Layout) -----------
with tab_info:
    st.header("🌐 Place Search")
    
    # Search box
    place_query = st.text_input(
        "Enter any Indian location (e.g., 'Taj Mahal', 'Hampi', 'Hyderabad'):",
        "",
        key="place_search"
    )
    
    if st.button("Search", key="search_btn"):
        if place_query:
            with st.spinner(f"Fetching information for {place_query}..."):
                summary, image_url, full_url = fetch_wikipedia_content(
                    place_query, 
                    lang=language_codes[selected_language]
                )

                if summary:
                    st.subheader(f"✨ {place_query}")
                    
                    # Horizontal layout container
                    col1, col2 = st.columns([1, 2])  # Image takes 1/3, text takes 2/3
                    
                    with col1:
                        if image_url:
                            try:
                                image_response = requests.get(image_url, headers=WIKIMEDIA_HEADERS, timeout=REQUEST_TIMEOUT)
                                image = Image.open(io.BytesIO(image_response.content))
                                st.image(image, use_column_width=True)
                            except Exception:
                                st.warning("Could not load image")
                    
                    with col2:
                        st.write(summary)
                        if full_url:
                            st.markdown(f"[Read full article on Wikipedia]({full_url})")
                else:
                    st.warning("No information found for this location.")

# ----------- Route Planner Tab (Compact Layout) -----------
with tab_route:
    st.header("🗺️ India Route Planner")
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        start_location = st.text_input("Start Location", "Hyderabad, Telangana")
    with col2:
        end_location = st.text_input("Destination", "Hampi, Karnataka")
    
    # Calculate route button
    if st.button("🚗 Calculate Route", type="primary"):
        if not start_location.strip() or not end_location.strip():
            st.error("Please enter both start and destination locations.")
        else:
            with st.spinner("Planning your route..."):
                try:
                    # Get coordinates
                    start_coords = get_coordinates(start_location)
                    end_coords = get_coordinates(end_location)
                    
                    if not start_coords:
                        st.error(f"Could not find coordinates for '{start_location}'")
                    elif not end_coords:
                        st.error(f"Could not find coordinates for '{end_location}'")
                    else:
                        # Get route
                        distance, duration, route_geometry = get_route(start_coords, end_coords)
                        
                        if None in [distance, duration, route_geometry]:
                            st.error("Could not calculate route between these locations.")
                        else:
                            st.session_state.route_calculated = True
                            st.session_state.route_data = {
                                'start_coords': start_coords,
                                'end_coords': end_coords,
                                'distance': distance,
                                'duration': duration,
                                'route_geometry': route_geometry,
                                'start_loc': start_location,
                                'end_loc': end_location
                            }
                            st.success("Route calculated successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    # Display route results if available
    if st.session_state.route_calculated and st.session_state.route_data:
        data = st.session_state.route_data
        
        # Route summary
        st.subheader("📍 Route Summary")
        cols = st.columns(3)
        cols[0].metric("Distance", f"{data['distance']:.1f} km")
        cols[1].metric("Duration", format_duration(data['duration']))
        avg_speed = data['distance'] / data['duration'] if data['duration'] > 0 else 0
        cols[2].metric("Avg Speed", f"{avg_speed:.0f} km/h")
        
        # Map display (reduced spacing)
        st.subheader("🗺️ Route Map")
        try:
            coords = polyline.decode(data['route_geometry'])
            m = folium.Map(
                location=[
                    (data['start_coords'][0] + data['end_coords'][0])/2,
                    (data['start_coords'][1] + data['end_coords'][1])/2
                ],
                zoom_start=7
            )
            
            # Add markers
            folium.Marker(
                data['start_coords'],
                tooltip=f"Start: {data['start_loc']}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            folium.Marker(
                data['end_coords'],
                tooltip=f"Destination: {data['end_loc']}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
            
            # Add route line
            folium.PolyLine(
                coords,
                color='blue',
                weight=4,
                opacity=0.8
            ).add_to(m)
            
            st_folium(m, width=700, height=400)
            
        except Exception as e:
            st.error(f"Could not display map: {str(e)}")
        
        # Add custom CSS to reduce margin before the next subheader
        st.write("""
            <style>
            /* This targets the div containing the subheader "Nearby Services" */
            /* and reduces its top margin */
            h2:has(span[data-testid="stMarkdownContainer"]:contains("Nearby Services")) {
                margin-top: -10px; /* Adjust this value as needed */
            }
            </style>
        """, unsafe_allow_html=True)

        # Nearby services (compact layout)
        st.subheader("📍 Nearby Services")
        
        # Use columns for better organization
        poi_cols = st.columns(3)
        
        with poi_cols[0]:
            st.markdown("**🍴 Food Places**")
            food_places = get_places_nearby(
                data['end_coords'][0],
                data['end_coords'][1],
                "food"
            )
            if food_places:
                for place in food_places[:5]:
                    st.markdown(f"- {place}")
            else:
                st.info("No food places found")
        
        with poi_cols[1]:
            st.markdown("**🏨 Hotels**")
            hotel_places = get_places_nearby(
                data['end_coords'][0],
                data['end_coords'][1],
                "hotels"
            )
            if hotel_places:
                for place in hotel_places[:5]:
                    st.markdown(f"- {place}")
            else:
                st.info("No hotels found")
        
        with poi_cols[2]:
            st.markdown("**⛽ Fuel Stations**")
            fuel_places = get_places_nearby(
                data['end_coords'][0],
                data['end_coords'][1],
                "fuel"
            )
            if fuel_places:
                for place in fuel_places[:5]:
                    st.markdown(f"- {place}")
            else:
                st.info("No fuel stations found")

# ----------- AI Chatbot Tab (Improved with Route Handling) -----------
with tab_chat:
    st.header("💬 AI Travel Guide")
    
    if not HF_API_KEY:
        st.warning("AI features require Hugging Face API key in your .env file")
    else:
        # Initialize chat history with a welcome message
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Namaste! I'm your India travel guide. You can ask me:\n- 'Route from Hyderabad to Hampi'\n- 'Tell me about Taj Mahal'\n- 'Hotels near Goa'\n\nHow can I help you today?"
            }]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about Indian places or routes..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                lower_prompt = prompt.lower()
                
                # Step 1: Check for route queries first
                if is_route_query(lower_prompt):
                    start, end = extract_route_locations(prompt)
                    
                    if start and end:
                        with st.spinner(f"Finding route from {start} to {end}..."):
                            # Use your existing route calculation functions
                            start_coords = get_coordinates(start)
                            end_coords = get_coordinates(end)
                            
                            if start_coords and end_coords:
                                distance, duration, geometry = get_route(start_coords, end_coords)
                                
                                if distance and duration:
                                    response = f"""🚗 **Route from {start.title()} to {end.title()}**:
                                    
- Distance: {distance:.1f} km
- Duration: {format_duration(duration)}
- Avg Speed: {distance/max(duration, 0.1):.0f} km/h
Switch to the 'Route Planner' tab for detailed map and nearby services!"""
                                    message_placeholder.markdown(response)
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response
                                    })
                                    st.stop()
                                else:
                                    response = f"I couldn't calculate the route between {start} and {end}. Please check the place names and try again."
                            else:
                                response = f"I couldn't find locations for {start} or {end}. Please try more specific names."
                            
                            message_placeholder.markdown(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            st.stop()
                
                # Step 2: Check for simple responses
                SIMPLE_RESPONSES = {
                    "hello": "Namaste! How can I help with your India travel plans today?",
                    "hi": "Namaste! Ask me about Indian destinations, culture, or travel tips!",
                    "help": "I can help with:\n- Route planning between cities\n- Information about places\n- Travel tips\n\nTry asking like: 'Route from Delhi to Jaipur' or 'Tell me about Goa beaches'",
                    "thank you": "You're welcome! Happy to help with your India travel journey.",
                }
                
                for question in SIMPLE_RESPONSES:
                    if question in lower_prompt:
                        message_placeholder.markdown(SIMPLE_RESPONSES[question])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": SIMPLE_RESPONSES[question]
                        })
                        st.stop()
                
                # Step 3: Try Wikipedia for factual place queries
                if any(keyword in lower_prompt for keyword in ["what is", "tell me about", "where is", "about"]):
                    place = extract_location_from_prompt(lower_prompt)
                    if place:
                        with st.spinner(f"Searching information about {place}..."):
                            summary, _, url = fetch_wikipedia_content(place.capitalize(), lang="en")
                            if summary:
                                response = format_wikipedia_response(place, summary, url)
                                message_placeholder.markdown(response)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response
                                })
                                st.stop()
                
                # Step 4: Check basic location info
                for city in BASIC_LOCATION_INFO:
                    if city in lower_prompt:
                        message_placeholder.markdown(BASIC_LOCATION_INFO[city])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": BASIC_LOCATION_INFO[city]
                        })
                        st.stop()
                
                # Step 5: Try Hugging Face API
                with st.spinner("Researching your question..."):
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 300,
                            "temperature": 0.7
                        }
                    }
                    
                    # Try with exponential backoff (3 attempts)
                    hf_response = None
                    for attempt in range(3):
                        try:
                            # Try different models in order of preference
                            for model in AVAILABLE_MODELS:
                                api_response = query_huggingface_api(payload, model)
                                if api_response and isinstance(api_response, list):
                                    if api_response[0].get('generated_text'):
                                        hf_response = api_response[0]['generated_text']
                                        break
                            
                            if hf_response:
                                break
                                
                            # If all models failed, wait longer and try again
                            wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                            time.sleep(wait_time)
                            
                        except Exception as e:
                            st.error(f"Error during attempt {attempt + 1}: {str(e)}")
                            time.sleep(1)
                    
                    if hf_response:
                        response = hf_response
                    else:
                        # Final fallback response
                        response = """I'm currently helping many travelers with their India trips. 
                        Here are some things I can tell you about:
                        
                        - Popular routes: Delhi to Jaipur, Mumbai to Goa, Hyderabad to Hampi
                        - Must-visit places: Taj Mahal, Kerala backwaters, Rajasthan forts
                        - Travel tips: Best seasons, local customs, transportation options
                        
                        Try asking about a specific route or destination!"""
                
                # Display the response
                message_placeholder.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })

# Footer
st.caption("Powered by Wikimedia APIs, OpenStreetMap services, and Hugging Face Inference API")
