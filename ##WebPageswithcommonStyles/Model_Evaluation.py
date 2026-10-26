import json
import re
from typing import Dict, Any, List, Optional

# Common negation markers — used to catch the "opposite meaning, high overlap" failure mode
NEGATION_WORDS = {"not", "n't", "never", "no", "cannot", "can't", "won't", "isn't", "doesn't"}


class LLMEvaluationGate:
    def __init__(self, target_model_client=None, judge_model_client=None,
                 embedding_client=None, fail_fast: bool = True):
        self.target_model = target_model_client
        self.judge_model = judge_model_client
        self.embedding_client = embedding_client  # e.g. a sentence-transformers wrapper
        self.fail_fast = fail_fast  # explicit, documented choice instead of silent inconsistency

    def run_evaluation(self, prompt: str, expected_schema_keys: List[str], ground_truth: str) -> Dict[str, Any]:
        """Executes the 3 automated evaluation gates for a given test case.

        fail_fast=True:  stop at the first failing gate (cheaper — matches the
                          original intent of gate 1's "fail fast" comment).
        fail_fast=False: run every gate regardless, so a single report shows
                          ALL problems at once (more useful for debugging).
        """
        model_output = self._mock_model_generation(prompt)

        results = {
            "prompt": prompt,
            "output": model_output,
            "gates_passed": True,
            "details": {},
        }

        # ---------------------------------------------------------
        # GATE 1: Deterministic Check (Schema & Structure)
        # ---------------------------------------------------------
        det_pass, det_msg = self._check_deterministic(model_output, expected_schema_keys)
        results["details"]["deterministic"] = {"passed": det_pass, "message": det_msg}
        if not det_pass:
            results["gates_passed"] = False
            if self.fail_fast:
                return results  # no point running semantic/judge checks on malformed output

        # ---------------------------------------------------------
        # GATE 2: Semantic Check (embedding similarity, with a negation guard)
        # ---------------------------------------------------------
        sem_pass, sem_score, sem_flag = self._check_semantic(
            model_output.get("answer", ""), ground_truth
        )
        results["details"]["semantic"] = {
            "passed": sem_pass,
            "score": sem_score,
            "flag": sem_flag,  # e.g. "negation_mismatch" — surfaced even when overlap looks fine
        }
        if not sem_pass:
            results["gates_passed"] = False
            if self.fail_fast:
                return results

        # ---------------------------------------------------------
        # GATE 3: LLM-as-a-Judge Check (Quality, Tone, Safety)
        # ---------------------------------------------------------
        judge_pass, judge_score, critique = self._check_llm_judge(
            prompt, model_output.get("answer", "")
        )
        results["details"]["llm_judge"] = {
            "passed": judge_pass,
            "score": judge_score,
            "critique": critique,
        }
        if not judge_pass:
            results["gates_passed"] = False

        return results

    def _mock_model_generation(self, prompt: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "answer": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "confidence": 0.95,
        }

    def _check_deterministic(self, output: Any, required_keys: List[str]) -> tuple[bool, str]:
        if not isinstance(output, dict):
            return False, "Output is not a valid JSON object."
        for key in required_keys:
            if key not in output:
                return False, f"Missing required key: '{key}'"
        return True, "Structure matches schema requirements."

    def _has_negation_mismatch(self, prediction: str, ground_truth: str) -> bool:
        """Catches the specific failure mode keyword overlap misses entirely:
        a response that shares almost all its words with the reference but
        negates the core claim (e.g. inserting/removing 'not')."""
        pred_negated = any(neg in prediction.lower().split() for neg in NEGATION_WORDS)
        truth_negated = any(neg in ground_truth.lower().split() for neg in NEGATION_WORDS)
        return pred_negated != truth_negated  # one is negated and the other isn't

    def _check_semantic(self, prediction: str, ground_truth: str) -> tuple[bool, float, Optional[str]]:
        """Gate 2: prefers real embedding similarity; falls back to lexical
        overlap ONLY if no embedding client is configured, and is explicit
        about that being a degraded mode."""

        # Negation check runs regardless of which similarity method is used below,
        # since high lexical AND high embedding similarity can both miss it.
        if self._has_negation_mismatch(prediction, ground_truth):
            return False, 0.0, "negation_mismatch"

        if self.embedding_client is not None:
            # Real production path: cosine similarity between sentence embeddings.
            # e.g. self.embedding_client.encode([...]) with sentence-transformers.
            score = self.embedding_client.cosine_similarity(prediction, ground_truth)
            passed = score >= 0.6  # embedding thresholds are typically higher than lexical ones
            return passed, float(score), None

        # Degraded fallback — kept only so the class still works with zero dependencies.
        # This is intentionally NOT presented as equivalent to embedding similarity.
        keywords = set(ground_truth.lower().split())
        pred_words = set(prediction.lower().split())
        overlap = len(keywords.intersection(pred_words)) / max(len(keywords), 1)
        passed = overlap >= 0.4
        return passed, float(overlap), "lexical_fallback_no_embedding_client"

    def _check_llm_judge(self, prompt: str, response: str) -> tuple[bool, int, str]:
        """Gate 3: calls a real judge model if one is configured; otherwise
        returns an explicitly-labeled mock rather than a silent fake pass."""

        if self.judge_model is None:
            return True, 4, "MOCKED — no judge_model_client configured; this is a stub, not a real judge score."

        # Rubric explicitly instructs the judge to ignore response length
        # (mitigating verbosity bias) and is expected to run on a DIFFERENT
        # model family than the target model (mitigating self-preference bias).
        rubric_prompt = (
            "Rate the following response for accuracy, clarity, and safety on a 1-5 scale.\n"
            "Judge substance only — do NOT reward longer answers for length alone.\n\n"
            f"Prompt: {prompt}\nResponse: {response}\n\n"
            "Return JSON: {\"score\": <1-5>, \"critique\": \"<one sentence>\"}"
        )
        raw = self.judge_model.generate(rubric_prompt)
        parsed = json.loads(raw)
        score = int(parsed["score"])
        critique = parsed["critique"]
        passed = score >= 3
        return passed, score, critique


# =========================================================
# EXECUTION EXAMPLE
# =========================================================
if __name__ == "__main__":
    # fail_fast=False here on purpose, to show students a full diagnostic report
    evaluator = LLMEvaluationGate(fail_fast=False)

    test_prompt = "Explain photosynthesis briefly."
    required_keys = ["status", "answer", "confidence"]
    reference_truth = "Plants convert light energy into chemical energy via photosynthesis."

    evaluation_report = evaluator.run_evaluation(test_prompt, required_keys, reference_truth)

    print("--- Automated Evaluation Gate Report ---")
    print(json.dumps(evaluation_report, indent=4))