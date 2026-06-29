import ast
import os
import pickle
import math
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from asttokens import ASTTokens


def get_node_key(node):
    """Map an AST node to a (name, category) key for DF counting."""
    t = type(node)
    if t is ast.Name:
        return (node.id, "Name")
    if t is ast.FunctionDef:
        return (node.name, "FunctionDef")
    if t is ast.AsyncFunctionDef:
        return (node.name, "AsyncFunctionDef")
    if t is ast.ClassDef:
        return (node.name, "ClassDef")
    if t is ast.Call:
        f = node.func
        ft = type(f)
        if ft is ast.Name:
            return (f.id, "Call")
        if ft is ast.Attribute:
            return (f.attr, "Call.Attribute")
        return ("Call", "Call")
    if t is ast.Attribute:
        return (node.attr, "Attribute")
    if t is ast.Subscript:
        s = node.slice
        if type(s) is ast.Index:
            s = s.value
        st = type(s)
        if st is ast.Name:
            return (s.id, "Subscript")
        if st is ast.Constant:
            return (str(s.value), "Subscript")
        return ("Subscript", "Subscript")
    if t is ast.List:
        return ("List", "List")
    if t is ast.Dict:
        return ("Dict", "Dict")
    if t is ast.Tuple:
        return ("Tuple", "Tuple")
    if t is ast.Constant:
        return (str(node.value), "Constant")
    return None


def _process_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        if code.count("\n") + 1 > 1000:
            return None
        tree = ast.parse(code)
        file_nodes = defaultdict(int)
        for node in ast.walk(tree):
            key = get_node_key(node)
            if key:
                file_nodes[key] += 1
        return filepath, dict(file_nodes)
    except Exception:
        return None


def calculate_df(data_dir, cache_file):
    """Build document-frequency stats from a directory of .py files (cached)."""
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".py"):
                all_files.append(os.path.join(root, file))
    node_stats = defaultdict(lambda: {"files": set(), "total_count": 0})
    total_files = 0
    with Pool(8) as pool:
        results = list(tqdm(pool.imap(_process_file, all_files), total=len(all_files), desc="Building DF stats"))
    for result in results:
        if result:
            filepath, file_nodes = result
            total_files += 1
            for key, count in file_nodes.items():
                node_stats[key]["files"].add(filepath)
                node_stats[key]["total_count"] += count
    result = {
        "node_stats": {k: {"df_obs": len(v["files"]), "total_count": v["total_count"]} for k, v in node_stats.items()},
        "total_files": total_files
    }
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def compute_idf_score(df_obs, total_files, max_df_obs, min_df_obs, eps=1e-12):
    """Compute normalized IDF score in [0, 1] with exponential transform."""
    df = max(df_obs / total_files, eps)
    df_common = max(max_df_obs / total_files, eps)
    s = -math.log(df)
    s_common = -math.log(df_common)
    df_rare_obs = max(min_df_obs / total_files, eps)
    s_rare_obs = -math.log(df_rare_obs)
    normalized = (s - s_common) / max(s_rare_obs - s_common, eps)
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    distance_to_top = 1.0 - normalized
    exp_neg_one = math.exp(-1.0)
    return (math.exp(-distance_to_top) - exp_neg_one) / (1.0 - exp_neg_one)


def inverse_document_frequency(code, cache_data):
    """Compute per-node IDF scores for a given code snippet using cached DF stats."""
    node_stats = cache_data["node_stats"]
    total_files = cache_data["total_files"]
    max_df_obs = max(v["df_obs"] for v in node_stats.values())
    min_df_obs = min(v["df_obs"] for v in node_stats.values())
    tree = ast.parse(code)
    code_nodes = set()
    for node in ast.walk(tree):
        key = get_node_key(node)
        if key is not None:
            code_nodes.add(key)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            code_nodes.add((node.arg, "Name"))
    idf_scores = {}
    for key in code_nodes:
        df_obs = node_stats[key]["df_obs"] if key in node_stats else 1
        idf_scores[key] = compute_idf_score(df_obs, total_files, max_df_obs, min_df_obs)
    return idf_scores


def _get_node_literal_range(node, atok):
    try:
        if isinstance(node, ast.Name):
            return atok.get_text_range(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            full_start, _ = atok.get_text_range(node)
            idx = atok.text.find(node.name, full_start)
            if idx != -1:
                return idx, idx + len(node.name)
        if isinstance(node, ast.Attribute):
            full_start, full_end = atok.get_text_range(node)
            idx = atok.text.rfind(node.attr, full_start, full_end)
            if idx != -1:
                return idx, idx + len(node.attr)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return atok.get_text_range(node.func)
            if isinstance(node.func, ast.Attribute):
                return _get_node_literal_range(node.func, atok)
        if isinstance(node, ast.Constant):
            return atok.get_text_range(node)
    except Exception:
        pass
    return None, None


def build_idf_node_index(code, cache_data, base_close_keywords=None, full_close_symbols=None):
    """Build sorted (abs_start, abs_end, idf, key_str) index for a code snippet."""
    if base_close_keywords is None:
        base_close_keywords = set()
    if full_close_symbols is None:
        full_close_symbols = set()
    idf_scores = inverse_document_frequency(code, cache_data)
    atok = ASTTokens(code, parse=True)
    tree = atok.tree
    lines = code.split("\n")
    line_positions = [0]
    for i in range(len(lines)):
        line_positions.append(line_positions[-1] + len(lines[i]) + 1)
    node_index = []
    for node in ast.walk(tree):
        key = get_node_key(node)
        if key is None or key == (None, None) or key not in idf_scores:
            continue
        node_start, node_end = _get_node_literal_range(node, atok)
        if node_start is None:
            continue
        node_text = code[node_start:node_end]
        has_keyword_or_symbol = node_text in base_close_keywords or node_text in full_close_symbols
        node_idf = 0.0 if has_keyword_or_symbol else idf_scores[key]
        node_index.append((node_start, node_end, node_idf, str(key)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.arg):
            continue
        key = (node.arg, "Name")
        if key not in idf_scores:
            continue
        if not (hasattr(node, "lineno") and hasattr(node, "col_offset")):
            continue
        node_start = line_positions[node.lineno - 1] + node.col_offset
        node_end = node_start + len(node.arg)
        node_text = code[node_start:node_end]
        has_keyword_or_symbol = node_text in base_close_keywords or node_text in full_close_symbols
        node_idf = 0.0 if has_keyword_or_symbol else idf_scores[key]
        node_index.append((node_start, node_end, node_idf, str(key)))
    node_index.sort(key=lambda x: x[0])
    return node_index


def idf_score_for_span(node_index, token_start, token_end, token_bare, base_close_keywords=None, full_close_symbols=None):
    """Return the max IDF score overlapping a given character span."""
    if base_close_keywords is None:
        base_close_keywords = set()
    if full_close_symbols is None:
        full_close_symbols = set()
    idf_weight = 0.0
    idf_source = []
    if token_bare in base_close_keywords or token_bare in full_close_symbols:
        return idf_weight, idf_source
    for node_start, node_end, node_idf, key_str in node_index:
        if node_start >= token_end:
            break
        if node_end <= token_start:
            continue
        overlap_start = max(token_start, node_start)
        overlap_end = min(token_end, node_end)
        if overlap_start < overlap_end and node_idf > idf_weight:
            idf_weight = node_idf
            idf_source = [key_str]
    return idf_weight, idf_source
