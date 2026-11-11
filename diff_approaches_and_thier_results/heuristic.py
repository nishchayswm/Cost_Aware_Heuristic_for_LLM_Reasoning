from sklearn.feature_extraction.text import TfidfVectorizer

# Minimum cutoff value for Reasoning Heuristic
e = 0.05 

def R(step_json):
    conf = step_json.get("confidence", 0.5)
    try:
        return max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        return 0.5

def T(c_texts, g_text):

    docs = c_texts + [g_text]
    vec = TfidfVectorizer().fit_transform(docs)
    goal_vec = vec[-1]
    cand_vecs = vec[:-1]
    sims = (cand_vecs @ goal_vec.T).toarray().flatten()
    lo, hi = sims.min(), sims.max()

    if hi - lo < 1e-9:
        return [0.5] * len(c_texts)
    
    return [(s - lo) / (hi - lo) for s in sims]

def C(depth, branch, d_max):
    remain = max(0, d_max - depth)
    return branch * remain

def final(R, T_norm, C, w1=0.5, w3=0.1):
    w2 = 1 - w1
    R = max(R, e)
    return R * (w1 + w2 * T_norm) / (1 + w3 * C)