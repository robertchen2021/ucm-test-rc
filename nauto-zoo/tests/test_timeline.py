from nauto_zoo.timeline import Timeline, TimelineElement, ModelResponseTimeline
import pytest


def test_should_identify_empty_timeline():
    assert Timeline("timeline_alias").is_empty()
    assert len(Timeline("timeline_alias")) == 0


def test_should_identify_non_empty_timeline():
    timeline = Timeline("timeline_alias")
    timeline.add_element(TimelineElement(start_ns=0, end_ns=1, element_type="name"))
    assert not timeline.is_empty()
    assert len(timeline) == 1


def test_should_expose_timeline_alias():
    timeline = Timeline("my_timeline_alias")
    timeline.add_element(TimelineElement(start_ns=0, end_ns=1, element_type="item_name"))
    assert timeline.review_type == "my_timeline_alias"


def test_raises_on_unreasonable_time_range():
    with pytest.raises(RuntimeError) as excinfo:
        timeline = Timeline("timeline_alias")
        timeline.add_element(TimelineElement(start_ns=2, end_ns=1, element_type="some_name"))
    assert str(excinfo.value) == 'Impossible timeline element - it starts at 2 ns and ends at 1 ns'


def test_should_show_invalid_when_no_offset():
    assert not Timeline("timeline_alias").is_valid()


def test_should_show_valid_when_offset_in_constructor():
    assert Timeline("timeline_alias", 0).is_valid()


def test_should_show_valid_when_offset_set():
    timeline = Timeline("timeline_alias")
    timeline.set_offset_ns(0)
    assert timeline.is_valid()


def test_can_instantiate_timeline_response():
    ModelResponseTimeline(timeline=Timeline(review_type="timeline_alias", offset_ns=1), summary="summary")


def test_should_expose_elements_iterator():
    timeline = Timeline("timeline_alias")
    timeline.add_element(TimelineElement(start_ns=1, end_ns=2, element_type="some_name", extra_fields={"value": "first_val"}))
    timeline.add_element(TimelineElement(start_ns=3, end_ns=4, element_type="some_name"))
    assert not timeline.is_empty()
    assert len(timeline) == 2
    iterator = timeline.elements()
    element: TimelineElement = next(iterator)
    assert element.start_ns == 1
    assert element.end_ns == 2
    assert element.extra_fields["value"] == "first_val"
    element: TimelineElement = next(iterator)
    assert element.start_ns == 3
    assert element.end_ns == 4


def test_timeline_repr_nonempty():
    timeline = Timeline(review_type="timeline_alias", offset_ns=123)
    timeline.add_element(TimelineElement(start_ns=0, end_ns=1, element_type="some_name"))
    timeline.add_element(TimelineElement(start_ns=2, end_ns=3, element_type="some_name"))
    assert repr(timeline) == "Timeline `timeline_alias`. Offset: `123`, elements: [[0, 1], [2, 3]]"


def test_timeline_repr_empty():
    timeline = Timeline(review_type="timeline_alias", offset_ns=123)
    assert repr(timeline) == "Empty timeline `timeline_alias`"


def test_drt_format():
    timeline = Timeline(review_type="timeline_alias", offset_ns=123)
    element = TimelineElement(start_ns=0, end_ns=1, element_type="some_name")
    timeline.set_offset_ns(123)
    timeline.add_element(element)
    assert element.drt_format("foo") == {
        'review_source': 'model',
        'review_type': 'foo',
        'label': {'confidence': 100, 'element_type': 'some_name', 'range': [0, 1]},
    }
    assert timeline.drt_format() == {
        'offset_ns': 123,
        'elements': [{
            'review_source': 'model',
            'review_type': 'timeline_alias',
            'label': {'confidence': 100, 'element_type': 'some_name', 'range': [0, 1]},
        }],
    }
