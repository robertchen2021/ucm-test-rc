from nauto_zoo import ModelResponseCollection, ModelResponse
from nauto_zoo.timeline import Timeline, TimelineElement, ModelResponseTimeline


def test_should_expose_responses():
    collection = ModelResponseCollection()
    collection.add(model_response=ModelResponse(score=.55, confidence=88, summary="SUMMARY"), judgement_type="judgement")
    timeline = Timeline("timeline_alias", offset_ns=1)
    timeline.add_element(TimelineElement(start_ns=1, end_ns=2, element_type="some_name"))
    timeline.add_element(TimelineElement(start_ns=5, end_ns=8, element_type="some_name"))
    collection.add(
        model_response=ModelResponseTimeline(score=.22, confidence=33, summary="TIMELINE_SUMMARY", timeline=timeline),
        judgement_type="timeline_judgement"
    )
    iterator = iter(collection)
    element = next(iterator)
    assert element.score == .55
    assert element.confidence == 88
    assert element.summary == "SUMMARY"
    assert element.judgement_type == "judgement"
    assert element.judgement_subtype is None
    assert not element.is_timeline()
    element = next(iterator)
    assert element.score == .22
    assert element.confidence == 33
    assert element.summary == "TIMELINE_SUMMARY"
    assert element.judgement_type == "timeline_judgement"
    assert element.judgement_subtype is None
    assert element.is_timeline()
    assert not element.timeline.is_empty()
    assert len(element.timeline) == 2


def test_should_expose_no_responses():
    collection = ModelResponseCollection()
    counter = 0
    for response in collection:
        counter += 1
    assert counter == 0


def test_should_expose_responses_with_judgement_subtype():
    collection = ModelResponseCollection()
    collection.add(model_response=ModelResponse(score=.1, confidence=1, summary="bar"), judgement_type="foo", judgement_subtype="sub")
    iterator = iter(collection)
    element = next(iterator)
    assert element.score == .1
    assert element.confidence == 1
    assert element.summary == "bar"
    assert element.judgement_type == "foo"
    assert element.judgement_subtype == "sub"
    assert not element.is_timeline()
