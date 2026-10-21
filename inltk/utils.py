import shutil
from pathlib import Path

import numpy as np


# cosine similarity
def cos_sim(v1, v2) -> float:
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        return 0.0
    return float(np.dot(v1, v2) / denominator)


def reset_models(folder_name: str):
    path = Path(__file__).parent
    shutil.rmtree(path / 'models' / f'{folder_name}')
    return


def is_english(s: str) -> bool:
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def handle_all_caps(t: str) -> str:
    from fastai.text.transform import replace_all_caps
    tokens = t.split()
    tokens = replace_all_caps(tokens)
    return ' '.join(tokens)


def handle_upper_case_first_letter(t: str) -> str:
    from fastai.text.transform import deal_caps
    tokens = t.split()
    tokens = deal_caps(tokens)
    return ' '.join(tokens)


def lower_case_everything(t: str) -> str:
    return t.lower()
