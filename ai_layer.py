import anthropic
import json
import requests
from datetime import datetime

client = anthropic.Anthropic()

EXTRACTION_PROMPT = """You are a data extraction assistant for Porter, 
a food delivery platform. Extract delivery order features from the 
user's natural language description.

Return ONLY a valid JSON object with these exact fields:
{{
    "created_at": "YYYY-MM-DD HH:MM:SS",
    "market_id": <int 1-6>,
    "store_primary_category": "<cuisine type>",
    "order_protocol": <int 1-7>,
    "total_items": <int>,
    "subtotal": <float in rupees>,
    "num_distinct_items": <int>,
    "min_item_price": <float>,
    "max_item_price": <float>,
    "total_onshift_partners": <float>,
    "total_busy_partners": <float>,
    "total_outstanding_orders": <float>,
    "missing_fields": ["list any fields you had to estimate"]
}}

Rules:
- Use current datetime if not specified: {current_time}
- If market not specified, use 3 (default)
- If order_protocol not specified, use 1 (default)
- If partner counts not mentioned, estimate based on time of day:
  * Peak hours (11-14, 17-21): onshift=25, busy=18, outstanding=30
  * Off peak: onshift=15, busy=8, outstanding=12
  * Late night: onshift=5, busy=3, outstanding=5
- If prices not specified but subtotal is, estimate min/max from subtotal
- Always populate missing_fields with any field you estimated
- Return ONLY the JSON, no explanation, no markdown backticks

User description: {user_input}"""

EXPLANATION_PROMPT = """You are a helpful customer service assistant 
for Porter food delivery. 

A customer's order has been analyzed. Explain the delivery prediction 
in a friendly, conversational way.

Order details:
{order_details}

Prediction:
- Estimated delivery: {predicted_mins} minutes
- Optimistic: {optimistic_mins} minutes  
- Pessimistic: {pessimistic_mins} minutes

Fields that were estimated (not provided by user):
{missing_fields}

Instructions:
- Be friendly and conversational, 2-3 sentences max
- Mention the time range naturally
- If fields were estimated, briefly mention what assumptions were made
- Don't use technical jargon like 'RMSE' or 'model prediction'
- End with something reassuring"""

def extract_features(user_input: str) -> dict:
    """
    Use Claude to extract structured features from natural language.
    Returns parsed JSON dict or raises ValueError if extraction fails.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(
                current_time=current_time,
                user_input=user_input
            )
        }]
    )
    
    response_text = message.content[0].text.strip()
    
    # Parse JSON — if this fails, extraction failed
    try:
        features = json.loads(response_text)
        return features
    except json.JSONDecodeError:
        # Try to extract JSON if wrapped in any text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse JSON from: {response_text}")

def get_prediction(features: dict) -> dict:
    """
    Call our FastAPI endpoint with extracted features.
    """
    # Remove our internal tracking field before sending to API
    payload = {k: v for k, v in features.items() 
               if k != 'missing_fields'}
    
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )
    
    if response.status_code != 200:
        raise ValueError(f"API error: {response.text}")
    
    return response.json()

def explain_prediction(features: dict, prediction: dict) -> str:
    """
    Use Claude to explain the prediction in plain English.
    """
    missing = features.get('missing_fields', [])
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user", 
            "content": EXPLANATION_PROMPT.format(
                order_details=json.dumps(features, indent=2),
                predicted_mins=prediction['predicted_delivery_minutes'],
                optimistic_mins=prediction['optimistic_minutes'],
                pessimistic_mins=prediction['pessimistic_minutes'],
                missing_fields=missing if missing else "None — all provided"
            )
        }]
    )
    
    return message.content[0].text.strip()

def full_pipeline(user_input: str) -> dict:
    """
    End to end: natural language → prediction → explanation
    """
    # Step 1: Extract features
    features = extract_features(user_input)
    missing = features.get('missing_fields', [])
    
    # Step 2: Get prediction from model
    prediction = get_prediction(features)
    
    # Step 3: Generate explanation
    explanation = explain_prediction(features, prediction)
    
    return {
        'explanation': explanation,
        'prediction': prediction,
        'extracted_features': features,
        'missing_fields': missing
    }