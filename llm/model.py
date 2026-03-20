import os
import groq
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT

load_dotenv()
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

ERROR_MESSAGES = {
    groq.RateLimitError:      "⚠️ Rate limit hit — wait before retrying.",
    groq.AuthenticationError: "❌ Invalid API key — check your .env file.",
    groq.APIConnectionError:  "🔌 Network issue — could not reach Groq.",
}

def analyze_negative_reviews(reviews: list[str]) -> str | None:
    formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews))
    prompt = f"Analyze these negative reviews and provide solutions:\n\n{formatted}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            timeout=20.0,
        )
        return response.choices[0].message.content

    except groq.APIStatusError as e:
        print(f"🏛️ Groq server error ({e.status_code}): {e.message}")
    except tuple(ERROR_MESSAGES) as e:
        print(ERROR_MESSAGES[type(e)])
    except Exception as e:
        print(f"❓ Unexpected error: {e}")

    return None
   
