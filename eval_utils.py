import pandas as pd
import time

class ResultsLogger:
    def __init__(self):
        self.results = []

    def log(self, agent_name, prompt, student_message, raw_output, parsed_state=None, error=None, **extra):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent_name,
            "student_message": student_message,
            "raw_output": raw_output,
            "valid": parsed_state is not None,
            "parsed_state": parsed_state.model_dump() if parsed_state else None,
            "error": str(error) if error else None,
        }
        entry.update(extra)  # room for judge_score, pre_test, post_test later in the week
        self.results.append(entry)

    def to_df(self):
        return pd.DataFrame(self.results)

    def fidelity_pct(self, agent_name=None):
        df = self.to_df()
        if agent_name:
            df = df[df["agent"] == agent_name]
        if len(df) == 0:
            return None
        return df["valid"].mean() * 100

    def save(self, path):
        self.to_df().to_csv(path, index=False)
        print(f"Saved {len(self.results)} results to {path}")