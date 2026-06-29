"""vLLM completions API wrappers for batch generation and echo logprob."""
import requests


def call_vllm_api(base_url, prompt, max_tokens=1, logprobs=None, temperature=0.0, timeout=300):
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    if logprobs is not None:
        payload["logprobs"] = logprobs
    try:
        response = requests.post(f"{base_url}/v1/completions", json=payload, timeout=timeout)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}"}
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def generate_text_batch_api(base_url, prefix_list, max_tokens=128, temperature=0.0, timeout=300):
    """Batch generate continuations, truncating at first newline."""
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prefix_list, "max_tokens": max_tokens, "temperature": temperature},
            timeout=timeout
        )
        if response.status_code != 200:
            return [""] * len(prefix_list)
        data = response.json()
        if "choices" not in data or not data["choices"]:
            return [""] * len(prefix_list)
        results = []
        for choice in data["choices"]:
            text = choice.get("text", "")
            nl_pos = text.find("\n")
            if nl_pos != -1:
                text = text[:nl_pos + 1]
            results.append(text.rstrip('\n'))
        return results
    except Exception:
        return [""] * len(prefix_list)


def get_echo_logprob_batch_api(base_url, prompt_list, timeout=300):
    """Batch echo with logprobs=1 to get token-level log-probabilities."""
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt_list, "max_tokens": 1, "logprobs": 1, "echo": True, "temperature": 0.0},
            timeout=timeout
        )
        if response.status_code != 200:
            return [None] * len(prompt_list)
        data = response.json()
        if "choices" not in data or not data["choices"]:
            return [None] * len(prompt_list)
        results = []
        for choice in data["choices"]:
            lp = choice.get("logprobs")
            results.append(lp.get("token_logprobs") if lp else None)
        return results
    except Exception:
        return [None] * len(prompt_list)
