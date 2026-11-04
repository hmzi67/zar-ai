from zarai.fusion.build_prompt import build_fusion_messages
from zarai.fusion.pipeline import CLARIFY, Pipeline
from zarai.nlp.baseline_translate.pipeline import translate_answer
from zarai.nlp.safety import DISCLAIMER, OFF_TOPIC_REFUSAL

CHUNK = {"chunk_id": "c1", "text": "Spray mancozeb.", "score": 1.0}


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return self.chunks


class FakeLLM:
    def __init__(self, text="Spray mancozeb."):
        self.text = text
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.text


class FakeClassifier:
    def __init__(self, abstain):
        self.result = {"top": [{"label": "leaf_rust", "prob": 0.7}], "abstain": abstain}

    def classify(self, image):
        return self.result


def test_prompt_contains_vision_question_and_chunks():
    messages = build_fusion_messages("q?", [{"label": "rust", "prob": 0.7}], [CHUNK], "en")
    assert messages[0]["role"] == "system"
    assert "rust: 0.70" in messages[1]["content"]
    assert "[c1] Spray mancozeb." in messages[1]["content"]
    assert "q?" in messages[1]["content"]


def test_abstaining_classifier_asks_to_clarify_without_calling_llm():
    llm = FakeLLM()
    pipe = Pipeline(retriever=FakeRetriever([CHUNK]), llm=llm, classifier=FakeClassifier(True))
    result = pipe.answer("What is this?", b"img")
    assert result["status"] == "clarify" and result["answer"] == CLARIFY["en"]
    assert llm.messages is None


def test_confident_classifier_feeds_retrieval_and_prompt():
    retriever, llm = FakeRetriever([CHUNK]), FakeLLM()
    pipe = Pipeline(retriever=retriever, llm=llm, classifier=FakeClassifier(False))
    result = pipe.answer("What is this?", b"img")
    assert result["status"] == "answered" and result["sources"] == ["c1"]
    assert result["answer"].endswith(DISCLAIMER["en"])
    assert "leaf_rust" in retriever.queries[0] and "leaf_rust" in llm.messages[1]["content"]


def test_no_chunks_refuses_in_question_language():
    result = Pipeline(retriever=FakeRetriever([]), llm=FakeLLM()).answer("kya hai")
    assert result["status"] == "refused" and result["answer"] == OFF_TOPIC_REFUSAL["ur-Latn"]


def test_ungrounded_llm_answer_is_blocked():
    pipe = Pipeline(retriever=FakeRetriever([CHUNK]), llm=FakeLLM("Use 500 ml glyphosate."))
    result = pipe.answer("What should I spray?")
    assert result["status"] == "blocked" and "500" not in result["answer"]


def test_translate_baseline_round_trip():
    class Translator:
        def translate(self, text, source, target):
            return f"{source}>{target}:{text}"

    retriever, llm = FakeRetriever([CHUNK]), FakeLLM("Spray mancozeb.")
    answer = translate_answer("گندم؟", Translator(), retriever, llm)
    assert retriever.queries == ["ur>en:گندم؟"]
    assert answer.startswith("en>ur:Spray mancozeb.")
