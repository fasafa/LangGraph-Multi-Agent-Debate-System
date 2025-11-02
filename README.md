# Multi-Agent Debate System (LangGraph)

This project simulates a structured **multi-agent debate** between two AI personas — a *Scientist* and a *Philosopher* — over a given topic using a Directed Acyclic Graph (DAG) flow built with LangGraph-style logic.

---

##  Features
- Two debating agents (Scientist & Philosopher)
- Turn-based dialogue using Qwen model
- Memory and Judge nodes for scoring and decision
- Debate transcript and logs
- Auto-generated DAG diagram of the debate flow

---

##  Project Structure
| File | Description |
|------|--------------|
| `main.py` | Main controller script |
| `node.py` | Node definitions for agents, memory, and judge |
| `qwen_utils.py` | Helper for Qwen model responses |
| `debate_log.txt` | Transcript of debate |
| `final_judgment.txt` | Declared winner and justification |
| `debate_graph.dot` | DAG structure file |
| `README.md` | Documentation file |


---

## How to Run
##### 1. Clone or unzip the project:
   ```bash
   git clone <your_repo_link>
   cd LangGraph-Multi-Agent-Debate-System
   ```


##### 2. Create environment and install dependencies:
      ```
      pip install -r requirements.txt

     ```

##### 3.Run the debate:
   ```
    python main.py
   ```
##### 4. Enter any topic when prompted (e.g., Should AI be regulated like medicine?)

---

##  DAG Diagram
<img width="619" height="513" alt="Image" src="https://github.com/user-attachments/assets/4dfae674-dd7a-4c6f-8e00-348ea0f98765" />

---

## Output Files
debate_log.txt → Transcript of debate

final_judgment.txt → Winner & reasoning

debate_graph.dot → Raw DAG graph




   

