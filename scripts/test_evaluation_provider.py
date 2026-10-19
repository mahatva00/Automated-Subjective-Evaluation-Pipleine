from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.static_answer_evaluator import GeminiClient, GroqClient, OllamaClient

def main():
    p=argparse.ArgumentParser(description="Test the chosen LLM provider using synthetic, non-student data.")
    p.add_argument("--provider", choices=("ollama","gemini","groq"), required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--timeout", type=int, default=90)
    a=p.parse_args()
    model=a.model or {"gemini":"gemini-3.8-flash","groq":"openai/gpt-oss-20b","ollama":"qwen2.5-coder:7b"}[a.provider]
    if a.provider=="gemini": client=GeminiClient(model=model,timeout=a.timeout)
    elif a.provider=="groq": client=GroqClient(model=model,timeout=a.timeout)
    else: client=OllamaClient(a.base_url,model,a.timeout)
    print(f"Testing {a.provider}/{model} with synthetic data only...")
    result=client.chat_json({"requirements_to_assess":[{"requirement_id":"TEST-1","requirement_text":"Return a valid assessment JSON for this synthetic test.","context_complete":False,"evidence_blocks":[]}]})
    if not isinstance(result,dict) or not isinstance(result.get("assessments"),list):
        raise SystemExit("Provider returned JSON but did not return an assessments array")
    print("Provider test passed: received valid JSON with an assessments array.")
    print("No student data was sent.")
    return 0
if __name__=="__main__": raise SystemExit(main())
