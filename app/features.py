"""Turn a raw bill into the columns the model was trained on.

Same logic as feature_engineering.ipynb (sections 1 and 4). If you change one, change the other.
"""

import html
import re

import numpy as np
import pandas as pd

ABBR = {"fb": "facebook", "ig": "instagram", "cdn": "content delivery network",
        "sms": "short message service", "atm": "cash", "sem": "search engine marketing"}


def clean(s):
    s = html.unescape(s.lower())                # "&amp;" -> "&"
    s = re.sub(r"^\s*\d{4}\b[\s\-_:]*", "", s)  # drop "0825 " month prefix
    words = re.findall(r"[a-z]+", s)            # letters only, numbers gone
    return " ".join(ABBR.get(w, w) for w in words)


def make_text(name, desc):
    name, desc = clean(name), clean(desc)
    if desc and not (name.startswith(desc) or desc.startswith(name)):
        return f"{name} {desc}".strip()
    return max(name, desc, key=len) or "nodescription"


def build_features(bills):
    """bills: list of dicts with vendorId, itemName, itemDescription, itemTotalAmount."""
    df = pd.DataFrame(bills)
    amount = df["itemTotalAmount"].astype(float)
    capped = amount.clip(-1_000_000, 1_000_000)

    return pd.DataFrame({
        "text": [make_text(n, d or "") for n, d in zip(df["itemName"], df["itemDescription"])],
        "vendorId": df["vendorId"],
        "log_amount": np.sign(capped) * np.log10(1 + capped.abs()),
        "is_neg": (amount < 0).astype(int),
    })
