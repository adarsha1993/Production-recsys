
import sys
sys.path.insert(0, "../../..")
import pytest

def test_expand_rules_thriller():
    from notebooks.week2_retrieval.rag_module         import expand_query_rules
    result = expand_query_rules("dark thriller")
    assert len(result) > len("dark thriller")

def test_expand_rules_no_match():
    from notebooks.week2_retrieval.rag_module         import expand_query_rules
    q      = "a movie"
    result = expand_query_rules(q)
    assert q in result

def test_expand_longer_than_original():
    from notebooks.week2_retrieval.rag_module         import expand_query_rules
    q      = "sci-fi adventure"
    result = expand_query_rules(q)
    assert len(result) > len(q)

def test_expand_comedy():
    from notebooks.week2_retrieval.rag_module         import expand_query_rules
    result = expand_query_rules("funny comedy")
    assert "humor" in result.lower() or            "funny" in result.lower()
