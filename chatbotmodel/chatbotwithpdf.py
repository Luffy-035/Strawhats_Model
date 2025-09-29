from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
import json
import numpy as np
from dotenv import load_dotenv
import uvicorn
from io import BytesIO
import tempfile
import pdfplumber
import re

# Character.AI and Groq imports for chatbot
from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError
from groq import Groq
from fastembed import TextEmbedding

# PDF processing imports
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as PlatypusImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load environment variables
load_dotenv()
token = "8bffea7f61747077512e09269760e1db113b59e7"
character_id = "5pXTea64l3x-_I9n4saY01AXiCP6uGLITeRVX1fk94k"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Platform",
    description="Combined Medical AI Chatbot and Report Parser API",
    version="2.0.0"
)

# Set matplotlib and seaborn styles globally
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.2)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# Serious health keywords for analysis
SERIOUS_HEALTH_KEYWORDS = [
    # Cardiac/Cardiovascular
    "heart pain", "chest pain", "cardiac arrest", "heart attack", "myocardial infarction",
    "angina", "chest tightness", "heart palpitations", "irregular heartbeat", "cardiac arrhythmia",
    "chest pressure", "crushing chest pain", "radiating pain", "left arm pain with chest pain",
    "jaw pain with chest discomfort", "shortness of breath with chest pain",
    
    # Respiratory/Breathing
    "difficulty breathing", "shortness of breath", "cannot breathe", "gasping for air",
    "choking", "suffocating", "respiratory distress", "blue lips", "cyanosis",
    "wheezing severely", "pneumonia", "asthma attack", "respiratory failure",
    "pulmonary embolism", "collapsed lung", "pneumothorax",
    
    # Neurological
    "severe headache", "intense headache", "worst headache of life", "migraine severe",
    "sudden severe headache", "thunderclap headache", "stroke", "paralysis",
    "sudden weakness", "facial drooping", "slurred speech", "confusion",
    "loss of consciousness", "seizure", "convulsions", "fits", "epileptic attack",
    "dizziness severe", "fainting", "passing out", "unconscious", "not waking up",
    "memory loss sudden", "sudden blindness", "sudden hearing loss",
    
    # Gastrointestinal
    "severe abdominal pain", "intense stomach pain", "appendicitis", "peritonitis",
    "intestinal obstruction", "bowel obstruction", "severe nausea vomiting",
    "blood in vomit", "vomiting blood", "hematemesis", "severe diarrhea",
    "bloody stool", "rectal bleeding", "severe constipation", "abdominal distension",
    
    # Additional serious conditions
    "severe", "intense", "unbearable", "excruciating", "urgent", "emergency",
    "doctor immediately", "hospital now", "call ambulance", "life threatening",
    "critical condition", "deteriorating rapidly", "getting worse fast",
    "can't function", "debilitating", "incapacitating"
]

# ==================== MEDICAL CHATBOT SECTION ====================

# Pydantic models for API
class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"

class Specialist(BaseModel):
    doctor_id: str
    name: str
    specialization: str
    category: str
    phone: str
    experience: int

class Analysis(BaseModel):
    detected_conditions: List[str]
    is_serious: bool
    recommended_specialty: str
    explanation: str

class ModelResponse(BaseModel):
    response: str
    analysis: Analysis
    specialists: Optional[List[Specialist]] = None
    is_serious: bool

# Core doctor retriever
class DoctorRetriever:
    def __init__(self, dataset_path="dataset.json"):
        self.embedding_model = TextEmbedding()
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.texts = []
        for d in self.data:
            availability = d.get("availability", "")
            if isinstance(availability, list):
                avail_str = " ".join([str(item) for item in availability if isinstance(item, str)])
            else:
                avail_str = str(availability)
            parts = [
                str(d.get("specialization", "")),
                str(d.get("category", "")),
                str(d.get("qualifications", "")),
                avail_str,
                str(d.get("name", ""))
            ]
            self.texts.append(" ".join(parts))
        self.embeddings = np.array(list(self.embedding_model.embed(self.texts)))

    def search(self, query, top_k=3):
        query_emb = np.array(list(self.embedding_model.embed([query]))[0])
        sims = np.dot(self.embeddings, query_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        best_idx = np.argpartition(sims, -top_k)[-top_k:]
        best_doctors = [self.data[i] for i in best_idx]
        return best_doctors

# Core message censor
class MessageCensor:
    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client
        self.violation_indicators = [
            "kms", "kys", "kill myself", "kill yourself", "suicide", "suicidal",
            "self harm", "self-harm", "cut myself", "hurt myself", "end my life",
            "want to die", "wanna die", "better off dead", "take my own life",
            "harm myself", "overdose", "pills", "rope", "bridge", "jump off",
            "slit my wrists", "hang myself", "shoot myself", "stab myself",
            "worthless", "hopeless", "can't go on", "no point living"
        ]

    def needs_censoring(self, message: str) -> bool:
        message_lower = message.lower().strip()
        for indicator in self.violation_indicators:
            if indicator in message_lower:
                return True
        return False

    def censor_message(self, message: str) -> str:
        if not self.needs_censoring(message):
            return message
        censor_prompt = f"""
        You are a message paraphraser. Your job is to take a user message that might be filtered by Character.AI and rewrite it in a way that:
        1. PRESERVES the exact emotional content and meaning
        2. REMOVES explicit self-harm language that triggers AI filters
        3. KEEPS the same tone, urgency, and desperation if present
        4. MAINTAINS the user's actual feelings and mental state
        5. MAKES it conversational and natural for AI chat
        6. EXPRESSES the same pain/struggle without trigger words

        Examples of good paraphrasing:
        - "I want to kms" → "I'm feeling so hopeless and desperate right now"
        - "I should just kill myself" → "I feel like there's no way out of this pain"
        - "Maybe I should hurt myself" → "I'm having really dark thoughts and urges"

        Keep the emotional intensity but make it chat-safe. Don't minimize their feelings.
        Original message: "{message}"
        Provide only the paraphrased version, nothing else:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": censor_prompt}],
                temperature=0.1,
                max_tokens=150
            )
            censored = response.choices[0].message.content.strip()
            if (censored.startswith('"') and censored.endswith('"')) or (censored.startswith("'") and censored.endswith("'")):
                censored = censored[1:-1]
            return censored
        except Exception:
            return self.basic_censor_fallback(message)

    def basic_censor_fallback(self, message: str) -> str:
        replacements = {
            "kms": "feel so hopeless",
            "kys": "you're really struggling",
            "kill myself": "feel like giving up",
            "kill yourself": "you're in so much pain",
            "suicide": "ending the pain",
            "suicidal": "having dark thoughts",
            "self harm": "hurting inside",
            "self-harm": "hurting inside",
            "end my life": "can't take this anymore",
            "want to die": "feel hopeless",
            "wanna die": "feel hopeless",
            "harm myself": "feel terrible inside",
            "hurt myself": "feel so much pain",
            "worthless": "feel like I don't matter",
            "no point living": "can't see any hope"
        }
        censored = message.lower()
        for original, replacement in replacements.items():
            censored = censored.replace(original, replacement)
        if message.isupper():
            return censored.upper()
        elif message.istitle():
            return censored.title()
        else:
            return censored

class EnhancedChatBot:
    def __init__(self, token: str, character_id: str, groq_api_key: str):
        self.token = token
        self.character_id = character_id
        self.groq_client = Groq(api_key=groq_api_key)
        self.censor = MessageCensor(self.groq_client)
        self.client = None
        self.chat = None
        self.me = None
        self.doctor_retriever = DoctorRetriever("dataset.json")
        self.chat_histories = {}

    async def chat_response(self, user_message: str, session_id: str = "default"):
        # Ensure CharacterAI session
        if self.client is None or self.me is None or self.chat is None:
            self.client = await get_client(token=self.token)
            self.me = await self.client.account.fetch_me()
            self.chat, _ = await self.client.chat.create_chat(self.character_id)

        # Step 1: Detect language and style (incl. Hinglish)
        language, is_hinglish, style = await self.detect_language_and_style(user_message)

        # Step 2: Translate to English if needed
        english_message = await self.translate_to_english(user_message)

        # Step 3: Extract illness/specialty (analysis logic)
        illness, specialties = await self.extract_illness_and_specialty(english_message)
        is_serious = False
        detected_conditions = []
        recommended_specialty = "None"
        explanation = ""
        specialists = None

        if illness:
            detected_conditions = [illness]
        else:
            detected_conditions = ["Non-specific symptoms"]

        lower_illness = (illness or "").lower()
        is_serious = any(keyword in lower_illness for keyword in SERIOUS_HEALTH_KEYWORDS)

        if is_serious:
            recommended_specialty = specialties[0] if specialties else "General Medicine"
            explanation = f"The user is experiencing {illness}, which is a symptom that can indicate a life-threatening condition such as a heart attack."
            rag_query = ", ".join(specialties) if specialties else illness
            doctors = self.doctor_retriever.search(rag_query, top_k=3)
            specialists = [
                Specialist(
                    doctor_id=str(doc.get("doctor_id", "")),
                    name=str(doc.get("name", "")),
                    specialization=str(doc.get("specialization", "")),
                    category=str(doc.get("category", "")),
                    phone=str(doc.get("phone", "")),
                    experience=int(doc.get("experience", 0))
                )
                for doc in doctors
            ] if doctors else None
        else:
            recommended_specialty = "None"
            explanation = "The query lacks specific symptoms or severity indicators to determine the need for specialist care."
            specialists = None

        # Step 4: Censor message if needed
        paraphrased_message = self.censor.censor_message(english_message)

        # Step 5: Call Character AI model and collect English response
        answer = await self.client.chat.send_message(
            self.character_id,
            self.chat.chat_id,
            paraphrased_message,
            streaming=True
        )
        full_response = ""
        async for r in answer:
            full_response = r.get_primary_candidate().text

        # Step 6: Translate model response back to user language/style
        if is_hinglish:
            user_final_response = await self.convert_to_hinglish(full_response, style)
        elif language.lower() != "english":
            user_final_response = await self.translate_from_english(language, full_response)
        else:
            user_final_response = full_response

        # Step 7: Build return object
        analysis_obj = Analysis(
            detected_conditions=detected_conditions,
            is_serious=is_serious,
            recommended_specialty=recommended_specialty,
            explanation=explanation
        )
        response_obj = ModelResponse(
            response=user_final_response,
            analysis=analysis_obj,
            specialists=specialists,
            is_serious=is_serious
        )
        return response_obj

    # Helper methods (unchanged pipeline)
    async def detect_language_and_style(self, user_message: str):
        prompt = f"""
        Analyze the following text and determine:
        1. The primary language/style being used
        2. Whether it's Hinglish (Hindi words written in English script mixed with English)
        Text: "{user_message}"
        Respond in JSON format:
        {{
            "language": "...",
            "is_hinglish": true/false,
            "style": "..."
        }}
        Examples:
        - "Hello kaise ho?" → {{"language": "Hinglish", "is_hinglish": true, "style": "Hindi-English mix"}}
        - "मैं ठीक हूँ" → {{"language": "Hindi", "is_hinglish": false, "style": "Pure Hindi"}}
        - "I am fine" → {{"language": "English", "is_hinglish": false, "style": "Pure English"}}
        Provide only the JSON, no extra text.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data.get("language", "English"), data.get("is_hinglish", False), data.get("style", "")
        except Exception:
            return "English", False, ""

    async def translate_to_english(self, user_message: str):
        prompt = f"""
        You are a multi-lingual assistant. Translate the following message to proper English.
        Handle Hinglish, Hindi, and other Indian languages appropriately.
        Input message: "{user_message}"
        Provide ONLY the English translation, nothing else.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            translation = response.choices[0].message.content.strip()
            if (translation.startswith('"') and translation.endswith('"')) or (translation.startswith("'") and translation.endswith("'")):
                translation = translation[1:-1]
            return translation
        except Exception:
            return user_message

    async def convert_to_hinglish(self, english_response: str, user_style: str):
        prompt = f"""
        You are a Hinglish conversion expert. Convert the following English text to Hinglish style that matches the user's original style.
        User's original style: {user_style}
        English text to convert: "{english_response}"
        Hinglish conversion guidelines:
        - Mix Hindi words (written in English) with English naturally
        - Keep the conversational tone
        - Use common Hindi words like: kya, hai, nahi, acha, bas, etc.
        - Examples: "How are you?" → "Kaise ho?" or "Aap kaise hain?"
        - "I am fine" → "Main theek hun" or "I'm fine yaar"
        Provide ONLY the Hinglish conversion, nothing else.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            hinglish_text = response.choices[0].message.content.strip()
            if (hinglish_text.startswith('"') and hinglish_text.endswith('"')) or (hinglish_text.startswith("'") and hinglish_text.endswith("'")):
                hinglish_text = hinglish_text[1:-1]
            return hinglish_text
        except Exception:
            return english_response

    async def translate_from_english(self, original_lang: str, english_response: str):
        prompt = f"""
        You are a multi-lingual assistant. Translate the following English text to {original_lang} language.
        English text: "{english_response}"
        Provide ONLY the translated text, nothing else.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            translation = response.choices[0].message.content.strip()
            if (translation.startswith('"') and translation.endswith('"')) or (translation.startswith("'") and translation.endswith("'")):
                translation = translation[1:-1]
            return translation
        except Exception:
            return english_response

    async def extract_illness_and_specialty(self, user_message: str):
        prompt = f"""
        You are a medical assistant. Given the user's message below, 
        1. Identify the main illness or symptoms described.
        2. List the doctor specialties appropriate for treating those symptoms.
        User message: "{user_message}"
        Respond in JSON format:
        {{
            "illness": "...",
            "specialties": ["..."] 
        }}
        Provide only the JSON, no extra text.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            illness = data.get("illness", "")
            specialties = data.get("specialties", [])
            return illness, specialties
        except Exception:
            return None, []

# ==================== PDF REPORT PARSER SECTION ====================

class PDFReportProcessor:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
    
    def get_report_type(self, report):
        """Enhanced report type detection"""
        report_type = report.get("report_type", "").lower()
        test_names = [test.get("test_name", "").lower() for test in report.get("test_results", [])]
        
        if any(keyword in report_type for keyword in ["blood", "hematology", "serum", "plasma", "cbc", "lipid", "glucose"]) or \
           any(test_name in ["hemoglobin", "wbc count", "rbc count", "platelet count", "glucose", "cholesterol"] 
               for test_name in test_names):
            return "blood"
        
        elif any(keyword in report_type for keyword in ["urine", "urinalysis", "ua"]) or \
             any(test_name in ["urine color", "urine ph", "specific gravity", "leukocytes", "nitrite", "protein", "glucose in urine", "ketones"] 
                 for test_name in test_names):
            return "urine"
        
        elif any(keyword in report_type for keyword in ["x-ray", "xray", "mri", "ct scan", "ultrasound", "imaging", "radiograph", "sonogram"]) or \
             any("impression" in test_name or "finding" in test_name for test_name in test_names):
            return "imaging"
        
        elif any(keyword in report_type for keyword in ["pathology", "histology", "biopsy", "cytology"]) or \
             any("specimen" in test_name or "tissue" in test_name for test_name in test_names):
            return "pathology"
        
        else:
            return "other"
    
    def parse_value_with_units(self, value_str):
        """Enhanced value parsing"""
        if not value_str:
            return None
        clean_str = value_str.replace(",", "")
        match = re.search(r'(\d+\.?\d*)', clean_str)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None
    
    def parse_range(self, range_str):
        """Enhanced range parsing"""
        if not range_str:
            return None, None
        
        if range_str.startswith('&amp;lt;') or range_str.startswith('&lt;'):
            try:
                clean_str = range_str.replace('&amp;lt;', '').replace('&lt;', '').strip()
                max_val = self.parse_value_with_units(clean_str)
                return 0, max_val
            except:
                return None, None
        elif range_str.startswith('&amp;gt;') or range_str.startswith('&gt;'):
            try:
                clean_str = range_str.replace('&amp;gt;', '').replace('&gt;', '').strip()
                min_val = self.parse_value_with_units(clean_str)
                return min_val, min_val * 2
            except:
                return None, None
        
        range_str = range_str.replace("–", "-").replace("—", "-").replace(" - ", "-")
        if "-" in range_str:
            parts = range_str.split("-")
            if len(parts) >= 2:
                min_val = self.parse_value_with_units(parts[0].strip())
                max_val = self.parse_value_with_units(parts[1].strip())
                return min_val, max_val
        
        return None, None
    
    def add_blood_test_bargraph(self, report, elements):
        """Complete blood test visualization from original code"""
        test_names = []
        actual_values = []
        normal_mins = []
        normal_maxs = []
        original_values = []
        original_ranges = []

        for test in report.get("test_results", []):
            test_name = test.get("test_name", "").strip()
            value_str = str(test.get("value", "")) if not isinstance(test.get("value"), float) else str(test.get("value", ""))
            ref_str = str(test.get("reference_range", ""))

            actual_val = self.parse_value_with_units(value_str)
            if actual_val is None:
                continue

            ref_min, ref_max = self.parse_range(ref_str)
            if ref_min is None or ref_max is None:
                continue

            test_names.append(test_name)
            actual_values.append(actual_val)
            normal_mins.append(ref_min)
            normal_maxs.append(ref_max)
            original_values.append(value_str)
            original_ranges.append(ref_str)

        if not test_names:
            elements.append(Paragraph("<b>Note:</b> No valid numerical data found for blood test chart generation.",
                                    getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 10))
            return

        max_tests_per_chart = 6
        num_charts = (len(test_names) + max_tests_per_chart - 1) // max_tests_per_chart

        for chart_index in range(num_charts):
            start_idx = chart_index * max_tests_per_chart
            end_idx = min(start_idx + max_tests_per_chart, len(test_names))

            chart_test_names = test_names[start_idx:end_idx]
            chart_actual_values = actual_values[start_idx:end_idx]
            chart_normal_mins = normal_mins[start_idx:end_idx]
            chart_normal_maxs = normal_maxs[start_idx:end_idx]

            # Create stunning visualization with dual subplot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

            # Determine colors based on normal range
            colors_list = []
            for actual, min_val, max_val in zip(chart_actual_values, chart_normal_mins, chart_normal_maxs):
                if actual < min_val:
                    colors_list.append('#FF6B6B')  # Red for below normal
                elif actual > max_val:
                    colors_list.append('#FF9F43')  # Orange for above normal  
                else:
                    colors_list.append('#26de81')  # Green for normal

            # Create the main bar plot
            bars = ax1.bar(chart_test_names, chart_actual_values, 
                          color=colors_list, edgecolor='white', linewidth=2,
                          alpha=0.9, width=0.6)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(chart_actual_values)*0.01,
                        f'{chart_actual_values[i]:.1f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11)

            # Add normal range as error bars with enhanced styling
            normal_midpoints = [(min_val + max_val) / 2 for min_val, max_val in zip(chart_normal_mins, chart_normal_maxs)]
            range_errors = [[(mid - min_val), (max_val - mid)] for mid, min_val, max_val in
                           zip(normal_midpoints, chart_normal_mins, chart_normal_maxs)]
            range_errors = np.array(range_errors).T

            ax1.errorbar(range(len(chart_test_names)), normal_midpoints, yerr=range_errors,
                        fmt='D', color='#2C3E50', capsize=8, capthick=3, 
                        markersize=8, markerfacecolor='white', markeredgewidth=2,
                        alpha=0.8, label='Normal Range')

            # Enhanced styling for main plot
            ax1.set_title(f'Blood Test Results Analysis (Chart {chart_index + 1} of {num_charts})', 
                         fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
            ax1.set_ylabel('Test Values', fontsize=14, fontweight='bold', color='#2C3E50')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_facecolor('#FAFAFA')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#E8E8E8')
            ax1.spines['bottom'].set_color('#E8E8E8')

            # Rotate x-axis labels for better readability
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='#26de81', alpha=0.9, label='Normal'),
                plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.9, label='Below Normal'),
                plt.Rectangle((0,0),1,1, facecolor='#FF9F43', alpha=0.9, label='Above Normal'),
            ]
            ax1.legend(handles=legend_elements, loc='upper right', frameon=True, 
                      facecolor='white', edgecolor='#E8E8E8')

            # Create status indicator subplot
            status_colors = []
            status_labels = []
            for i, (actual, min_val, max_val, test_name) in enumerate(zip(
                chart_actual_values, chart_normal_mins, chart_normal_maxs, chart_test_names)):
                if actual < min_val:
                    status_colors.append('#FF6B6B')
                    status_labels.append('↓ Low')
                elif actual > max_val:
                    status_colors.append('#FF9F43') 
                    status_labels.append('↑ High')
                else:
                    status_colors.append('#26de81')
                    status_labels.append('✓ Normal')

            # Status indicator chart
            ax2.barh(chart_test_names, [1]*len(chart_test_names), color=status_colors, alpha=0.8)
            
            for i, (label, color) in enumerate(zip(status_labels, status_colors)):
                ax2.text(0.5, i, label, ha='center', va='center', 
                        fontweight='bold', color='white')

            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Status Indicator', fontsize=12, fontweight='bold', color='#2C3E50')
            ax2.set_facecolor('#FAFAFA')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xticks([])

            plt.tight_layout()

            img_buffer = BytesIO()
            plt.savefig(img_buffer, format="png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            img_buffer.seek(0)

            elements.append(Spacer(1, 15))
            chart_heading = "Blood Test Analysis"
            if num_charts > 1:
                chart_heading += f" (Chart {chart_index + 1} of {num_charts})"

            elements.append(Paragraph(chart_heading, getSampleStyleSheet()['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(PlatypusImage(img_buffer, width=600, height=400))
            elements.append(Spacer(1, 20))

            # Enhanced comparison table with deviation calculation
            elements.append(Paragraph("Detailed Results:", getSampleStyleSheet()['Heading3']))
            elements.append(Spacer(1, 8))

            comparison_header = ["Test Name", "Your Value", "Normal Range", "Status", "Deviation"]
            comparison_rows = [comparison_header]

            for i, (test_name, original_val, original_range, actual_val, min_val, max_val) in enumerate(
                    zip(chart_test_names, original_values[start_idx:end_idx], original_ranges[start_idx:end_idx], 
                        chart_actual_values, chart_normal_mins, chart_normal_maxs)):

                if actual_val < min_val:
                    status = "Below Normal"
                    deviation = f"-{((min_val - actual_val) / min_val * 100):.1f}%"
                elif actual_val > max_val:
                    status = "Above Normal"
                    deviation = f"+{((actual_val - max_val) / max_val * 100):.1f}%"
                else:
                    status = "Normal"
                    deviation = "Within range"

                comparison_rows.append([test_name, original_val, original_range, status, deviation])

            comparison_table = Table(comparison_rows, colWidths=[100, 70, 90, 80, 80], hAlign="CENTER")
            comparison_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E8E8E8")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
            ]))

            for i in range(1, len(comparison_rows)):
                status = comparison_rows[i][3]
                if status == "Normal":
                    comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.HexColor("#A8E6CF"))]))
                elif status == "Below Normal":
                    comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.HexColor("#FFE5E5"))]))
                else:
                    comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.HexColor("#FFF4E5"))]))

            elements.append(comparison_table)
            elements.append(Spacer(1, 25))

    def add_urine_test_visualization(self, report, elements):
        """Complete urine test visualization with heatmap"""
        styles = getSampleStyleSheet()
        test_results = report.get("test_results", [])

        if not test_results:
            elements.append(Paragraph("<b>Note:</b> No urine test data found for visualization.", styles['Normal']))
            elements.append(Spacer(1, 10))
            return

        elements.append(Paragraph("<b>Urine Test Results:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))

        # Enhanced color mapping from Streamlit version
        color_mapping = {
            "color": {
                "normal": ["pale yellow", "yellow", "straw", "amber", "clear"],
                "abnormal": ["red", "brown", "orange", "green", "blue", "cloudy", "turbid"]
            },
            "clarity": {
                "normal": ["clear", "transparent"],
                "abnormal": ["cloudy", "turbid", "hazy"]
            },
            "ph": {
                "normal": lambda v: 4.5 <= self.parse_value_with_units(v) <= 8.0 if self.parse_value_with_units(v) is not None else False
            },
            "specific gravity": {
                "normal": lambda v: 1.005 <= self.parse_value_with_units(v) <= 1.030 if self.parse_value_with_units(v) is not None else False
            },
            "glucose": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
            },
            "protein": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
            },
            "ketones": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
            },
            "blood": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
            },
            "nitrite": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive"]
            },
            "leukocytes": {
                "normal": ["negative", "none", "0", "normal"],
                "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
            },
            "bacteria": {
                "normal": ["negative", "none", "0", "normal", "not seen"],
                "abnormal": ["positive", "present", "few", "moderate", "many"]
            },
            "epithelial cells": {
                "normal": ["negative", "none", "0", "normal", "few", "occasional"],
                "abnormal": ["moderate", "many"]
            }
        }

        header = ["Parameter", "Result", "Reference Range", "Status"]
        rows = [header]
        test_names = []
        statuses = []

        for test in report.get("test_results", []):
            test_name = (test.get("test_name") or "").strip().lower()
            value_raw = test.get("value") or ""
            value = str(value_raw).strip().lower() if not isinstance(value_raw, float) else str(value_raw).lower()
            ref_range = str(test.get("reference_range") or "").strip()

            if not test_name or not value:
                continue

            status = "Normal"
            status_numeric = 1

            for param, rules in color_mapping.items():
                if param in test_name:
                    if "normal" in rules and callable(rules["normal"]):
                        if not rules["normal"](value):
                            status = "Abnormal"
                            status_numeric = 0
                    elif "normal" in rules and isinstance(rules["normal"], list):
                        if not any(normal_val in value for normal_val in rules["normal"]) or \
                           any(abnormal_val in value for abnormal_val in rules.get("abnormal", [])):
                            status = "Abnormal"
                            status_numeric = 0
                    break
            else:
                if ref_range and value:
                    num_value = self.parse_value_with_units(value)
                    min_val, max_val = self.parse_range(ref_range)
                    if num_value is not None and min_val is not None and max_val is not None:
                        if num_value < min_val or num_value > max_val:
                            status = "Abnormal"
                            status_numeric = 0

            rows.append([test_name.title(), value, ref_range, status])
            test_names.append(test_name.title())
            statuses.append(status_numeric)

        # Create enhanced table
        if len(rows) > 1:
            table = Table(rows, colWidths=[100, 80, 100, 60], hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E86AB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E8E8E8")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
            ]))

            for i in range(1, len(rows)):
                status = rows[i][3]
                if status == "Normal":
                    table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.HexColor("#A8E6CF"))]))
                else:
                    table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.HexColor("#FFB3BA"))]))

            elements.append(table)
            elements.append(Spacer(1, 20))

        # Create stunning status heatmap
        if test_names and statuses:
            fig, ax = plt.subplots(figsize=(12, max(3, len(test_names) * 0.4)))
            
            df_heatmap = pd.DataFrame({
                'Test': test_names,
                'Status': statuses
            })
            
            pivot_data = df_heatmap.set_index('Test')['Status'].to_frame().T
            
            sns.heatmap(pivot_data, 
                       annot=False,
                       cmap=['#FFB3BA', '#A8E6CF'],
                       cbar_kws={'label': 'Test Status (Red=Abnormal, Green=Normal)'},
                       linewidths=2,
                       linecolor='white',
                       ax=ax)
            
            ax.set_title('Urine Test Status Overview', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('')
            ax.set_xlabel('Tests', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format="png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            img_buffer.seek(0)
            elements.append(Paragraph("<b>Urine Test Status Heatmap:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(PlatypusImage(img_buffer, width=550, height=max(150, len(test_names) * 20)))
            elements.append(Spacer(1, 20))

        # Add urine color visualization
        for test in test_results:
            if "color" in (test.get("test_name") or "").lower():
                color_value = (test.get("value") or "").lower()
                if color_value:
                    elements.append(Paragraph("<b>Urine Color Representation:</b>", styles['Heading3']))
                    elements.append(Spacer(1, 8))
                    color_hex = {
                        "pale yellow": "#FFFFA0",
                        "yellow": "#FFFF00",
                        "dark yellow": "#CCCC00",
                        "amber": "#FFBF00",
                        "orange": "#FFA500",
                        "red": "#FF0000",
                        "pink": "#FFC0CB",
                        "brown": "#A52A2A",
                        "clear": "#F0F8FF",
                        "cloudy": "#E6E6FA"
                    }
                    color_code = "#FFFFA0"
                    for color_name, hex_code in color_hex.items():
                        if color_name in color_value:
                            color_code = hex_code
                            break
                    color_table = Table([[""], [""], [""]], colWidths=[100], rowHeights=[50, 50, 50])
                    color_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (0, 2), colors.HexColor(color_code)),
                        ("BOX", (0, 0), (0, 2), 1, colors.black),
                    ]))
                    elements.append(color_table)
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"<i>Reported color: {color_value}</i>", styles['Normal']))
                    elements.append(Spacer(1, 20))
                    break

        def add_imaging_report_visualization(self, report, elements):
            styles = getSampleStyleSheet()
            findings = None
            impression = None

        for test in report.get("test_results", []):
            test_name = (test.get("test_name") or "").lower()
            if "finding" in test_name:
                findings = test.get("value")
            elif "impression" in test_name:
                impression = test.get("value")

        if not findings and not impression and "doctor_notes" in report:
            notes = report.get("doctor_notes", "")
            if "finding" in notes.lower():
                findings_section = re.search(r'(?i)findings?:(.+?)(?:impression:|assessment:|conclusion:|$)', notes, re.DOTALL)
                if findings_section:
                    findings = findings_section.group(1).strip()

            if "impression" in notes.lower():
                impression_section = re.search(r'(?i)impression:(.+?)(?:recommendation:|plan:|$)', notes, re.DOTALL)
                if impression_section:
                    impression = impression_section.group(1).strip()

        if findings:
            elements.append(Paragraph("<b>Findings:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(findings, styles['Normal']))
            elements.append(Spacer(1, 15))

        if impression:
            elements.append(Paragraph("<b>Impression:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(impression, styles['Normal']))
            elements.append(Spacer(1, 15))

        if not findings and not impression:
            elements.append(Paragraph("<b>Note:</b> No structured findings or impressions found in this imaging report.",
                                    styles['Normal']))
            elements.append(Spacer(1, 10))

    def add_pathology_report_visualization(self, report, elements):
        """Complete pathology report visualization"""
        styles = getSampleStyleSheet()
        specimen = None
        diagnosis = None
        microscopic = None

        for test in report.get("test_results", []):
            test_name = (test.get("test_name") or "").lower()
            if "specimen" in test_name:
                specimen = test.get("value")
            elif "diagnosis" in test_name:
                diagnosis = test.get("value")
            elif "microscopic" in test_name:
                microscopic = test.get("value")

        if not any([specimen, diagnosis, microscopic]) and "doctor_notes" in report:
            notes = report.get("doctor_notes", "")
            if "specimen" in notes.lower():
                specimen_section = re.search(r'(?i)specimen:(.+?)(?:clinical|gross|microscopic|diagnosis:|$)', notes, re.DOTALL)
                if specimen_section:
                    specimen = specimen_section.group(1).strip()

            if "diagnosis" in notes.lower():
                diagnosis_section = re.search(r'(?i)diagnosis:(.+?)(?:comment:|note:|$)', notes, re.DOTALL)
                if diagnosis_section:
                    diagnosis = diagnosis_section.group(1).strip()

            if "microscopic" in notes.lower():
                microscopic_section = re.search(r'(?i)microscopic:(.+?)(?:diagnosis:|assessment:|$)', notes, re.DOTALL)
                if microscopic_section:
                    microscopic = microscopic_section.group(1).strip()

        if specimen:
            elements.append(Paragraph("<b>Specimen:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(specimen, styles['Normal']))
            elements.append(Spacer(1, 15))

        if diagnosis:
            elements.append(Paragraph("<b>Diagnosis:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(diagnosis, styles['Normal']))
            elements.append(Spacer(1, 15))

        if microscopic:
            elements.append(Paragraph("<b>Microscopic Description:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(microscopic, styles['Normal']))
            elements.append(Spacer(1, 15))

        if not any([specimen, diagnosis, microscopic]):
            elements.append(Paragraph("<b>Note:</b> No structured pathology data found in this report.",
                                    styles['Normal']))
            elements.append(Spacer(1, 10))

    def add_generic_report_visualization(self, report, elements):
        """Complete generic report visualization"""
        styles = getSampleStyleSheet()
        elements.append(Paragraph("<b>Note:</b> This is a general medical report without specific visualization.",
                                styles['Normal']))
        elements.append(Spacer(1, 15))

        test_names = []
        values = []
        units = []
        ref_ranges = []

        for test in report.get("test_results", []):
            test_name = (test.get("test_name") or "").strip()
            value_raw = test.get("value") or ""
            value_str = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
            unit = str(test.get("unit") or "").strip()
            ref_range = str(test.get("reference_range") or "").strip()

            if test_name and value_str:
                test_names.append(test_name)
                values.append(value_str)
                units.append(unit)
                ref_ranges.append(ref_range)

        if test_names:
            elements.append(Paragraph("<b>Test Results Summary:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))

            header = ["Test", "Result", "Unit", "Reference Range"]
            rows = [header]

            for i in range(len(test_names)):
                rows.append([test_names[i], values[i], units[i], ref_ranges[i]])

            table = Table(rows, colWidths=[120, 100, 80, 120], hAlign="CENTER")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E8E8E8")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

    def extract_test_results(self, reports, format_type="dict"):
        """Extract test results functionality"""
        results = {}
        if not isinstance(reports, list):
            reports = [reports]

        for report in reports:
            if not report or "test_results" not in report:
                continue
            for test in report.get("test_results", []):
                test_name = test.get("test_name", "").strip()
                if not test_name:
                    continue
                value_raw = test.get("value", "")
                value = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
                unit = test.get("unit", "")
                if unit:
                    value = f"{value} {unit}"
                results[test_name] = value

        if format_type.lower() == "json":
            return json.dumps(results, indent=2)
        return results

    def merge_reports(self, reports):
        """Complete merge reports function"""
        if not reports:
            return []

        if len(reports) == 1:
            return reports

        merged_report = {
            "patient_info": {},
            "report_type": "Combined Medical Report",
            "test_results": [],
            "doctor_notes": "",
            "summary": ""
        }

        unique_tests = set()
        doctor_notes = []
        summaries = []

        # Use the first non-empty patient info
        for report in reports:
            if report.get("patient_info") and any(str(v).strip() for v in report["patient_info"].values() if v):
                merged_report["patient_info"] = report["patient_info"]
                break

        # Merge test results, avoiding duplicates
        for report in reports:
            if report.get("doctor_notes"):
                report_type = report.get("report_type", "Unknown")
                doctor_notes.append(f"[{report_type.upper()} REPORT] {report['doctor_notes']}")

            if report.get("summary"):
                report_type = report.get("report_type", "Unknown")
                summaries.append(f"[{report_type.upper()}] {report['summary']}")

            for test in report.get("test_results", []):
                test_name = str(test.get("test_name", "")).strip().lower()
                if not test_name or test_name in unique_tests:
                    continue

                unique_tests.add(test_name)
                merged_report["test_results"].append(test)

        merged_report["doctor_notes"] = "\n\n".join(doctor_notes)
        merged_report["summary"] = "\n\n".join(summaries)

        return [merged_report]

    def generate_pdf(self, parsed_reports, output_buffer):
        """Complete PDF generation with all visualization functions"""
        styles = getSampleStyleSheet()
        elements = []
        doc = SimpleDocTemplate(
            output_buffer,
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=60,
            bottomMargin=60
        )

        for i, report in enumerate(parsed_reports):
            elements.append(Paragraph("<b>Medical Report</b>", styles['Title']))
            elements.append(Spacer(1, 18))

            if "patient_info" in report:
                patient_info = report["patient_info"]
                pat_text = "   ".join([
                    f"Name: {patient_info.get('name', 'N/A')}",
                    f"Age: {patient_info.get('age', 'N/A')}",
                    f"Sex: {patient_info.get('sex', 'N/A')}"
                ])
                elements.append(Paragraph(pat_text, styles['Normal']))
                elements.append(Spacer(1, 18))

            elements.append(Paragraph(
                f"Report Type: {report.get('report_type','Unknown')}",
                styles['Heading2']
            ))
            elements.append(Spacer(1, 18))

            if "test_results" in report and report["test_results"]:
                header = ["Test Name", "Value", "Unit", "Reference Range"]
                all_rows = [header]

                for test in report["test_results"]:
                    test_name = (test.get("test_name") or "").strip()
                    value_raw = test.get("value") or ""
                    value = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
                    unit = str(test.get("unit") or "").strip()
                    ref_range = str(test.get("reference_range") or "").strip()

                    if not test_name or (not value and not unit and not ref_range):
                        continue

                    all_rows.append([test_name, value, unit, ref_range])

                if len(all_rows) > 1:
                    max_rows = 8
                    chunks = [all_rows[i:i + max_rows] for i in range(0, len(all_rows), 7)]

                    for idx, chunk in enumerate(chunks):
                        if idx > 0:
                            chunk.insert(0, header)

                        table = Table(chunk, colWidths=[120, 100, 80, 120], hAlign="CENTER")
                        header_color = colors.HexColor("#2E86AB") if idx % 2 == 0 else colors.HexColor("#A23B72")

                        table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), header_color),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 1), (-1, -1), 10),
                            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E8E8E8")),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ]))

                        elements.append(table)
                        elements.append(Spacer(1, 20))

            # Enhanced visualization selection
            report_type = self.get_report_type(report)
            if report_type == "blood":
                self.add_blood_test_bargraph(report, elements)
            elif report_type == "urine":
                self.add_urine_test_visualization(report, elements)
            elif report_type == "imaging":
                self.add_imaging_report_visualization(report, elements)
            elif report_type == "pathology":
                self.add_pathology_report_visualization(report, elements)
            else:
                self.add_generic_report_visualization(report, elements)

            if "doctor_notes" in report:
                elements.append(Paragraph("<b>Doctor Notes:</b>", styles['Heading3']))
                elements.append(Spacer(1, 8))
                elements.append(Paragraph(report["doctor_notes"], styles['Normal']))
                elements.append(Spacer(1, 25))

            if i < len(parsed_reports) - 1:
                elements.append(PageBreak())

        doc.build(elements)

# Global bot instance
bot_instance = None

@app.on_event("startup")
async def startup_event():
    global bot_instance
    if GROQ_API_KEY == "your_groq_api_key_here":
        raise RuntimeError("❌ ERROR: Please set your Groq API key in .env!")
    bot_instance = EnhancedChatBot(token, character_id, GROQ_API_KEY)
    await asyncio.sleep(0.01)

# Initialize PDF processor
pdf_processor = PDFReportProcessor()

# ==================== PYDANTIC MODELS ====================

# PDF Models
class PDFParseRequest(BaseModel):
    files: List[str] = []

# ==================== API ENDPOINTS ====================

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Medical AI Platform API is running!",
        "services": ["chatbot", "pdf_parser"],
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    global bot_instance
    return {
        "status": "healthy",
        "chatbot_status": bot_instance is not None,
        "pdf_processor_status": pdf_processor.groq_client is not None
    }

# ==================== CHATBOT ENDPOINTS ====================

@app.post("/chat", response_model=ModelResponse)
async def chat_endpoint(prompt_request: PromptRequest):
    global bot_instance
    if bot_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    try:
        return await bot_instance.chat_response(prompt_request.prompt, prompt_request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_chat")
async def clear_chat(session_id: str = "default"):
    """Clear chat history for a specific session"""
    global bot_instance
    try:
        if bot_instance and hasattr(bot_instance, 'chat_histories') and session_id in bot_instance.chat_histories:
            bot_instance.chat_histories[session_id] = []
        return {"message": f"Chat history cleared for session: {session_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat: {str(e)}")

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active chat sessions"""
    global bot_instance
    if bot_instance and hasattr(bot_instance, 'chat_histories'):
        return {"active_sessions": list(bot_instance.chat_histories.keys())}
    return {"active_sessions": []}

@app.post("/analyze", response_model=Analysis)
async def analyze_medical_query_endpoint(prompt_request: PromptRequest):
    """Analyze a medical query without generating a chat response"""
    global bot_instance
    if bot_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    try:
        # Extract illness and specialty using the bot's method
        illness, specialties = await bot_instance.extract_illness_and_specialty(prompt_request.prompt)
        
        detected_conditions = [illness] if illness else ["Non-specific symptoms"]
        lower_illness = (illness or "").lower()
        is_serious = any(keyword in lower_illness for keyword in SERIOUS_HEALTH_KEYWORDS)
        
        if is_serious:
            recommended_specialty = specialties[0] if specialties else "General Medicine"
            explanation = f"The user is experiencing {illness}, which may require specialist attention."
        else:
            recommended_specialty = "None"
            explanation = "The query lacks specific symptoms or severity indicators to determine specialist care need."
        
        return Analysis(
            detected_conditions=detected_conditions,
            is_serious=is_serious,
            recommended_specialty=recommended_specialty,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing query: {str(e)}")

# ==================== PDF PARSER ENDPOINTS ====================

@app.post("/parse_report/")
async def parse_report(files: List[UploadFile] = File(...)):
    """Enhanced parse_report endpoint with complete PDF processing"""
    # Validate that all files are PDFs
    for file in files:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be a PDF.")

    all_reports = []
    temp_files = []

    try:
        # Process each file
        for file in files:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_temp:
                pdf_temp.write(contents)
                pdf_temp.seek(0)
                temp_pdf_path = pdf_temp.name
                temp_files.append(temp_pdf_path)

            # Extract text from PDF
            with pdfplumber.open(temp_pdf_path) as pdf:
                pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            max_text_length = 2500
            truncated_pdf_text = pdf_text[:max_text_length] if len(pdf_text) > max_text_length else pdf_text

            system_prompt = """Extract structured medical data as JSON with STRICT formatting requirements:

CRITICAL: ALL values must be properly quoted strings, even if they contain numbers or ranges.

Schema:
{
  "patient_info": {"name": "string", "age": "string", "sex": "string"},
  "report_type": "string", 
  "test_results": [{"test_name": "string", "value": "string", "unit": "string", "reference_range": "string"}],
  "doctor_notes": "string",
  "summary": "string"
}

IMPORTANT RULES:
- ALL field values must be strings enclosed in double quotes
- For ranges like "0-2 /hpf", write as "0-2 /hpf" (quoted)
- For numbers like 1.020, write as "1.020" (quoted)  
- For numeric values like 6.0, write as "6.0" (quoted)
- Never use unquoted values like: 1.020 or 0-2 /hpf
- Always use quoted values like: "1.020" or "0-2 /hpf"

For the summary: Write 2-4 sentences covering test types, normal/abnormal values, and overall status."""

            max_retries = 3
            parsed_data = None
            
            for attempt in range(max_retries):
                try:
                    models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
                    model = models[min(attempt, len(models) - 1)]
                    
                    completion = pdf_processor.groq_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Parse this medical report into valid JSON with all values as quoted strings:\n\n{truncated_pdf_text}"}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    
                    raw_response = completion.choices[0].message.content
                    
                    # Basic JSON cleaning for common issues
                    cleaned_response = raw_response
                    
                    # Fix patterns like "value": 1.020, -> "value": "1.020",
                    cleaned_response = re.sub(r'"value":\s*(\d+\.?\d*),', r'"value": "\1",', cleaned_response)
                    # Fix patterns like "age": 32, -> "age": "32", 
                    cleaned_response = re.sub(r'"age":\s*(\d+),', r'"age": "\1",', cleaned_response)
                    # Fix range patterns like "value": 0-2 /hpf, -> "value": "0-2 /hpf",
                    cleaned_response = re.sub(r'"value":\s*([0-9\-\+\s\/\w]+),', r'"value": "\1",', cleaned_response)
                    
                    parsed_data = json.loads(cleaned_response)
                    
                    if isinstance(parsed_data, dict):
                        parsed_data = [parsed_data]
                    break
                    
                except json.JSONDecodeError as json_error:
                    print(f"JSON decode error on attempt {attempt + 1}: {json_error}")
                    if attempt == max_retries - 1:
                        parsed_data = [{
                            "patient_info": {"name": "Unknown", "age": "Unknown", "sex": "Unknown"},
                            "report_type": "Medical Report",
                            "test_results": [],
                            "doctor_notes": f"Error parsing report: {str(json_error)}",
                            "summary": "Unable to parse medical report due to formatting issues."
                        }]
                    continue
                except Exception as api_error:
                    print(f"API error on attempt {attempt + 1}: {api_error}")
                    if attempt == max_retries - 1:
                        raise api_error
                    continue
            
            all_reports.extend(parsed_data)

        # Merge reports if multiple files were uploaded
        if len(files) > 1:
            final_reports = pdf_processor.merge_reports(all_reports)
        else:
            final_reports = all_reports

        # Extract test results only
        test_results_only = pdf_processor.extract_test_results(final_reports)
        test_results_json = pdf_processor.extract_test_results(final_reports, format_type="json")

        # Generate PDF with complete visualization functions
        pdf_buffer = BytesIO()
        pdf_processor.generate_pdf(final_reports, pdf_buffer)
        pdf_buffer.seek(0)

        # Create output filename
        if len(files) == 1:
            output_pdf_name = f"{os.path.splitext(files[0].filename)[0]}_report.pdf"
        else:
            output_pdf_name = "combined_medical_report.pdf"
        
        output_pdf_path = os.path.join(tempfile.gettempdir(), output_pdf_name)
        
        pdf_content = pdf_buffer.read()
        
        with open(output_pdf_path, "wb") as f:
            f.write(pdf_content)
        
        if not os.path.exists(output_pdf_path):
            raise HTTPException(status_code=500, detail="Failed to create PDF file")
        
        file_size = os.path.getsize(output_pdf_path)
        print(f"✅ PDF created successfully: {output_pdf_path} (Size: {file_size} bytes)")

        return {
            "parsed_json": final_reports,
            "test_results_json": test_results_only,
            "pdf_download_url": f"/download_report/{output_pdf_name}",
            "total_files_processed": len(files),
            "total_reports_merged": len(all_reports),
            "unique_tests_found": len(final_reports[0].get('test_results', [])) if final_reports else 0,
            "pdf_file_size": file_size
        }

    except Exception as e:
        print(f"Full error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

@app.get("/download_report/{filename}")
def download_report(filename: str):
    """Download generated PDF report"""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type="application/pdf")

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
