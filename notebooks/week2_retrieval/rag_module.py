"""
RAG query expansion module.
Importable by other notebooks and tests.
"""
import requests

GENRE_EXPANSIONS = {
    "thriller":    "suspense tension psychological "
                   "danger mysterious dark atmosphere",
    "comedy":      "funny humor lighthearted amusing "
                   "entertaining witty laughter",
    "romance":     "love relationship emotional "
                   "heartwarming chemistry passion",
    "action":      "adventure excitement battle "
                   "heroic intense explosive",
    "horror":      "scary frightening dark terrifying "
                   "supernatural suspense",
    "sci-fi":      "science fiction future technology "
                   "space exploration dystopian",
    "science":     "science fiction future technology",
    "drama":       "emotional story character "
                   "development realistic life",
    "animated":    "animation colorful family "
                   "cartoon adventure imaginative",
    "animation":   "colorful family cartoon "
                   "adventure imaginative",
    "crime":       "detective mystery investigation "
                   "criminal justice noir",
    "war":         "battle military conflict "
                   "soldiers courage sacrifice",
    "fantasy":     "magic mythical creatures "
                   "adventure epic otherworldly",
    "biography":   "true story real person "
                   "historical inspiring life",
    "inception":   "mind-bending non-linear dream "
                   "psychological heist cerebral",
    "intense":     "gripping powerful dramatic "
                   "high-stakes tension",
    "funny":       "humorous comedy lighthearted "
                   "laugh entertaining",
    "family":      "suitable all ages wholesome "
                   "heartwarming children",
}

MOOD_EXPANSIONS = {
    "dark":       "atmospheric moody intense "
                  "gritty noir shadow",
    "light":      "bright cheerful uplifting "
                  "feel-good optimistic",
    "emotional":  "moving touching heartfelt "
                  "tear-jerking powerful",
    "exciting":   "thrilling fast-paced adrenaline "
                  "action-packed suspenseful",
    "thought":    "cerebral intellectual "
                  "philosophical complex",
}


def expand_query_rules(query: str) -> str:
    """Rule-based query expansion"""
    query_lower = query.lower()
    expansions  = [query]
    for keyword, expansion in {
        **GENRE_EXPANSIONS,
        **MOOD_EXPANSIONS
    }.items():
        if keyword in query_lower:
            expansions.append(expansion)
    return " ".join(expansions)


def expand_query_ollama(
        query: str,
        model: str = "llama3.2",
        host: str  = "localhost",
        port: int  = 11434) -> str:
    """Expand query using Ollama"""
    try:
        prompt = (
            f"Expand this movie search query "
            f"into a rich 2-sentence description "
            f"with mood, themes and genre: {query}"
        )
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150,
                }
            },
            timeout=60,
        )
        if response.status_code == 200:
            expanded = response.json()                .get("response", "").strip()
            if expanded and len(expanded) > 20:
                return f"{query}. {expanded}"
    except Exception:
        pass
    return expand_query_rules(query)
