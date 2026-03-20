import re

REFUSAL_TEXT = "i cannot answer this question based on the provided documents."


def is_refusal(answer: str) -> bool:
    return answer.strip().lower() == REFUSAL_TEXT


def _token_set(text: str):
    return set(re.findall(r"\b\w+\b", text.lower()))


def simple_similarity(a: str, b: str) -> float:
    a_words = _token_set(a)
    b_words = _token_set(b)

    if not a_words or not b_words:
        return 0.0

    return len(a_words & b_words) / len(a_words | b_words)


def calculate_answer_accuracy(answer, docs):
    if not docs:
        return 0.0

    context = " ".join(d.page_content for d in docs)
    sim = simple_similarity(answer, context)
    return round(sim * 100, 2)


def calculate_faithfulness(answer, docs):
    if not answer or not docs:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 5]
    if not sentences:
        return 0.0

    context = " ".join(d.page_content for d in docs)

    supported = 0
    for sent in sentences:
        sim = simple_similarity(sent, context)
        if sim > 0.20:
            supported += 1

    return round((supported / len(sentences)) * 100, 2)


def calculate_relevance(question, answer):
    sim = simple_similarity(question, answer)
    return round(sim * 100, 2)