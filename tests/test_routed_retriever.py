from zarai.nlp.retriever.hybrid import ScriptRoutedRetriever


class Named:
    def __init__(self, name):
        self.name = name

    def search(self, query, k=5):
        return [{"chunk_id": self.name, "text": query, "score": 1.0}][:k]


def test_urdu_script_goes_to_urdu_retriever():
    routed = ScriptRoutedRetriever(Named("urdu"), Named("latin"))
    assert routed.search("گندم کو پانی کب دیں؟")[0]["chunk_id"] == "urdu"


def test_roman_urdu_and_english_go_to_latin_retriever():
    routed = ScriptRoutedRetriever(Named("urdu"), Named("latin"))
    assert routed.search("gandum ko pani kab dein?")[0]["chunk_id"] == "latin"
    assert routed.search("When should I irrigate wheat?")[0]["chunk_id"] == "latin"


def test_k_is_passed_through():
    routed = ScriptRoutedRetriever(Named("urdu"), Named("latin"))
    assert routed.search("wheat", k=0) == []
