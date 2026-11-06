"""Deterministic perturbations with explicit edit/label-validity records."""
import hashlib
import re
import numpy as np
from .config import ValidationError

TOKEN = re.compile(r"\b[A-Za-z]+\b")
PROTECTED = {"not", "no", "nor", "without", "never", "none", "neither"}
NEIGHBORS = dict(zip("qwertyuiopasdfghjklzxcvbnm", ["wa","qes","wrd","etf","ryg","tuh","yij","uok","ipl","ok","qwsz","awed","serx","drtc","ftyv","gyub","huni","jiom","kop","asx","zsdc","xdfv","cfgb","vghn","bhjm","njk"]))

def make_query_typos(queries, severity=1, seed=42):
    if severity not in (0, 1, 2, 3):
        raise ValidationError("Supported typo severity: 0, 1, 2, 3. Use the declared cache grid.")
    result = {}
    for qid, text in queries.items():
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:16], 16))
        candidates = [m for m in TOKEN.finditer(text) if len(m.group()) >= 5
                      and m.group().islower() and m.group().lower() not in PROTECTED
                      and not (m.start() > 0 and (text[m.start()-1].isdigit() or text[m.start()-1] == '-'))
                      and not (m.end() < len(text) and (text[m.end()].isdigit() or text[m.end()] == '-'))]
        rng.shuffle(candidates)
        edits = []
        for m in candidates[:severity]:
            token = m.group()
            pos = int(rng.integers(1, len(token)-1))
            op = ["substitute", "insert", "delete"][int(rng.integers(0, 3))]
            near = NEIGHBORS[token[pos]]
            char = near[int(rng.integers(0, len(near)))]
            replacement = (token[:pos]+char+token[pos+1:] if op == "substitute" else
                           token[:pos]+char+token[pos:] if op == "insert" else token[:pos]+token[pos+1:])
            edits.append({"start": m.start(), "end": m.end(), "before": token, "after": replacement, "operation": op})
        changed = text
        for e in sorted(edits, key=lambda x: x["start"], reverse=True):
            changed = changed[:e["start"]] + e["after"] + changed[e["end"]:]
        result[str(qid)] = {"text": changed, "original": text, "edits": edits,
                            "edit_count": len(edits), "requested_edits": severity,
                            "flags": ["meaning-risk", "not-human-reviewed"] if edits else ["unchanged"],
                            "excluded_reason": "no eligible token" if severity and not edits else None,
                            "seed": seed, "severity": severity}
    return result

def perturbation_accounting(records):
    n = len(records)
    changed = sum(r["edit_count"] > 0 for r in records.values())
    full = sum(r["edit_count"] == r["requested_edits"] for r in records.values())
    return {"queries": n, "modified": changed, "unchanged": n-changed,
            "requested_edits_achieved": full, "mechanical_valid_rate": full/n if n else 0,
            "human_reviewed": sum("validity-reviewed" in r["flags"] for r in records.values()),
            "semantic_validity": "conditional; automatic edits are not a semantic guarantee"}
