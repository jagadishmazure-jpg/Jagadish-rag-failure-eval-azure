import pytest
from conftest import chunk

from ragfail import data
from ragfail.corpus import INJECTIONS
from ragfail.guards import MAX_QUESTION_CHARS, check_question, sentences, strip_instructions
from ragfail.reader import ABSTAIN, EXAMPLES, build_messages


@pytest.mark.parametrize("text,canary", INJECTIONS[:3])
def test_filter_removes_the_planted_instructions(text, canary):
    clean, n = strip_instructions(f"The fee is $640. {text} Quotes last ninety days.")
    assert n >= 1 and canary not in clean and "$640" in clean and "ninety days" in clean


def test_known_miss_politely_worded_injection_passes_the_filter():
    # KNOWN MISS: no "ignore" or "assistant" wording, so the regex filter keeps it. The system prompt and
    # the eval are the remaining defences; see README "Limitations".
    text, canary = INJECTIONS[3]
    clean, n = strip_instructions(text)
    assert n == 0 and canary in clean


def test_no_false_positives_on_clean_chunks():
    canaries = [c for _, c in INJECTIONS]
    for s in ("para", "w80", "doc"):
        for c in data.chunks(s):
            if not any(x in c["text"] for x in canaries):
                assert strip_instructions(c["titled"]) == (c["titled"], 0), c["chunk_id"]


def test_sentence_split_keeps_quoted_text_together():
    assert sentences('Reply only with "Stop. Pay now." Then go.') == ['Reply only with "Stop. Pay now."', "Then go."]


def test_question_checks():
    with pytest.raises(ValueError):
        check_question("   ")
    with pytest.raises(ValueError):
        check_question("x" * (MAX_QUESTION_CHARS + 1))


def test_guarded_prompt_strips_and_warns_unguarded_does_not():
    bad = chunk(text="The fee is $640. " + INJECTIONS[0][0], titled="T. The fee is $640. " + INJECTIONS[0][0])
    msgs, shown, n = build_messages("What is the fee?", [bad], guarded=True)
    assert n == 2 and INJECTIONS[0][1] not in shown[0] and "never follow instructions" in msgs[0]["content"]
    msgs, shown, n = build_messages("What is the fee?", [bad], guarded=False)
    assert n == 0 and INJECTIONS[0][1] in shown[0] and "never follow" not in msgs[0]["content"]


def test_worked_examples_leak_no_corpus_content():
    text = " ".join(d["text"] for d in data.corpus())
    for _u, a in EXAMPLES:
        assert "Garden tools" not in text and (a == ABSTAIN or a.split(" [")[0] not in text)
