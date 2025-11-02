
"""
Nodes for the debate:
- ScientistNode and PhilosopherNode (use Qwen generation helper)
- MemoryNode: stores transcript & agent summaries
- JudgeNode: inspects memory, scores, and declares winner
- Helper to export a simple DOT graph from the history
"""

from typing import Dict, Any, List
from datetime import datetime
import hashlib
import json

# Attempt to import the user's qwen helper (handles both filename spellings)
try:
    from qwen_utils import generate_qwen_reply
except Exception:
    try:
        from qwen_utlis import generate_qwen_reply
    except Exception:
        # If neither import works, provide a fallback dummy generator
        def generate_qwen_reply(prompt: str, max_new_tokens: int = 200) -> str:
            # deterministic fallback so program still runs even without model
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:6]
            return f"(fallback generated argument {h})"


# ----- Agent nodes -----
def generate_one_argument_with_qwen(prompt: str, max_new_tokens=120) -> str:
    """
    Uses external qwen helper to produce a reply.
    Includes fallback and retry logic for empty responses.
    """
    for attempt in range(3):  # retry up to 3 times
        try:
            raw = generate_qwen_reply(prompt, max_new_tokens=max_new_tokens)
            text = raw.strip()

            # Handle empty or garbage outputs
            if not text:
                print(f"[Warning] Empty response on attempt {attempt + 1}, retrying...")
                continue

            firstline = text.split("\n")[0]
            sentence = firstline.split(".")[0].strip()

            # Validate content
            if sentence:
                return sentence + "."
        except Exception as e:
            print(f"[Error] Generation failed on attempt {attempt + 1}: {e}")

    # Final fallback after retries
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:6]
    return f"(unable to generate valid response {h})"


class ScientistNode:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        last_opponent = state.get("last_philosopher") or "No prior argument."
        prompt = (
            f"Persona: Scientist\n"
            f"Topic: {state['topic']}\n"
            f"Opponent's argument: {last_opponent}\n"
            f"Produce one concise scientist-style argumentative sentence."
        )
        argument = generate_one_argument_with_qwen(prompt, max_new_tokens=120)
        state["last_scientist"] = argument
        state["history"].append(("Scientist", argument))
        return state


class PhilosopherNode:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        last_opponent = state.get("last_scientist") or "No prior argument."
        prompt = (
            f"Persona: Philosopher\n"
            f"Topic: {state['topic']}\n"
            f"Opponent's argument: {last_opponent}\n"
            f"Produce one concise philosopher-style argumentative sentence."
        )
        argument = generate_one_argument_with_qwen(prompt, max_new_tokens=120)
        state["last_philosopher"] = argument
        state["history"].append(("Philosopher", argument))
        return state


def create_initial_state(topic: str) -> Dict[str, Any]:
    return {
        "topic": topic,
        "last_scientist": None,
        "last_philosopher": None,
        "history": []
    }


# ----- Memory node -----
class MemoryNode:
    """
    Stores the full transcript in self.memory (list of dict entries)
    and offers simple per-agent summaries.
    """
    def __init__(self):
        self.memory: List[Dict[str, Any]] = []

    def update(self, entry: Dict[str, Any]):
        """entry: {round, speaker, argument, timestamp}"""
        self.memory.append(entry)

    def get_summary_for(self, speaker: str, max_len=400) -> str:
        parts = [m["argument"] for m in self.memory if m["speaker"] == speaker]
        summary = " | ".join(parts[-3:])
        if len(summary) > max_len:
            summary = summary[:max_len - 3] + "..."
        return summary


# ----- Judge node -----
class JudgeNode:
    """
    Evaluates memory and declares a winner. Uses heuristic scoring.
    """
    def __init__(self):
        pass

    def score_argument(self, arg_text: str, memory: List[Dict[str, Any]]) -> float:
        tokens = set(arg_text.lower().split())
        seen = set()
        for m in memory:
            seen |= set(m["argument"].lower().split())
        novelty = len(tokens - seen)
        length_score = max(1.0, min(10.0, len(arg_text.split()) / 5.0))
        repeat_penalty = 5.0 if sum(1 for m in memory if m["argument"] == arg_text) > 1 else 0.0
        return novelty + length_score - repeat_penalty

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        memory = context.get("memory", [])
        topic = context.get("topic", "")
        if not memory:
            return {"winner": "NoDebate", "justification": "No arguments were made.", "summary": ""}

        transcript_lines = [f"Round {m['round']} [{m['speaker']}]: {m['argument']}" for m in memory]
        full_summary = "\n".join(transcript_lines)

        scores = {"Scientist": 0.0, "Philosopher": 0.0}
        per_round_scores = []
        seen_tokens = set()
        for m in memory:
            arg = m["argument"]
            tokens = set(arg.lower().split())
            novelty = len(tokens - seen_tokens)
            length_score = max(1.0, min(10.0, len(arg.split()) / 5.0))
            repeat_penalty = 5.0 if any(prev["argument"] == arg for prev in memory if prev is not m) else 0.0
            score = novelty + length_score - repeat_penalty
            per_round_scores.append({"round": m["round"], "speaker": m["speaker"], "score": score})
            scores[m["speaker"]] += score
            seen_tokens |= tokens

        if scores["Scientist"] > scores["Philosopher"]:
            winner = "Scientist"
            reason = f"Scientist scored {scores['Scientist']:.2f} vs Philosopher {scores['Philosopher']:.2f} (higher novelty/content)."
        elif scores["Philosopher"] > scores["Scientist"]:
            winner = "Philosopher"
            reason = f"Philosopher scored {scores['Philosopher']:.2f} vs Scientist {scores['Scientist']:.2f} (higher novelty/content)."
        else:
            winner = "Tie"
            reason = f"Scores tied (Scientist {scores['Scientist']:.2f}, Philosopher {scores['Philosopher']:.2f}). Declaring a tie."

        return {
            "topic": topic,
            "scores": scores,
            "per_round_scores": per_round_scores,
            "winner": winner,
            "justification": reason,
            "summary": full_summary
        }


# ----- DOT graph helper -----
def export_graph_dot_from_history(memory: List[Dict[str, Any]], path: str = "debate_graph.dot") -> str:
    """
    Create a small DOT file that shows each round node and edges between rounds
    (simple linear DAG). This is a tiny LangGraph-like visual.
    """
    lines = ["digraph Debate {", '  rankdir=LR;']
    lines.append('  topic [label="User: Topic"];')
    prev_node = "topic"
    for m in memory:
        node_id = f'r{m["round"]}_{m["speaker"]}'
        label = f'R{m["round"]}\\n{m["speaker"]}'
        lines.append(f'  "{node_id}" [label="{label}"];')
        lines.append(f'  "{prev_node}" -> "{node_id}";')
        prev_node = node_id
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
