# WikiYatra

**WikiYatra** is a smart, multilingual travel assistant that helps users discover culturally rich destinations in India. The app integrates **Wikimedia APIs** for destination information, **Google Maps** for route planning, and an **AI-powered chatbot** for interactive travel guidance.

---

## Features

* Search for destinations and get detailed information in multiple languages
* Plan routes and view maps using Google Maps integration
* Interactive AI chatbot for travel recommendations and guidance
* Mobile-first and user-friendly interface
* Secure handling of API keys via `.env` file

---

## Tech Stack

* **Frontend:** Streamlit (or FastAPI + Streamlit)
* **Backend / AI:** Hugging Face API / AI chatbot
* **APIs:** Wikimedia API, Google Maps API
* **Language:** Python
* **Environment Variables:** `.env` file for API keys

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file**
   Copy `.env.example` and add your API keys:

```bash
cp .env.example .env
```

Update `.env`:

```
HUGGINGFACE_API_KEY=your_key_here
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## Folder Structure

```
WikiYatra/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md

```
