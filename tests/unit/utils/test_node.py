import pytest

from src.utils.node import *

def apex_node_b_content_id():
    return '456b'

@pytest.fixture(scope="session")
def apex_node_with_no_children_fixture():
    return all_nodes_fixture()['123a']

@pytest.fixture(scope="session")
def apex_node_with_children_fixture():
    return all_nodes_fixture()[apex_node_b_content_id()]

@pytest.fixture(scope="session")
def leaf_node_fixture():
    return all_nodes_fixture()['345b/b/a']

@pytest.fixture(scope="session")
def level_two_node_fixture():
    return all_nodes_fixture()['789b/a']

nodes = {}

def all_nodes_fixture():
    apex_node_a_fixture = {
        'base_path': 'a/',
        'content_id': '123a',
        'title': 'taxon a',
    }
    apex_node_b_fixture = {
        'base_path': 'b/',
        'content_id': apex_node_b_content_id(),
        'title': 'taxon b',
    }
    b_child_a_fixture = {
        'base_path': 'b/a',
        'content_id': '789b/a',
        'parent_content_id': apex_node_b_content_id(),
        'title': 'taxon b - child a',
    }
    b_child_b_fixture = {
        'base_path': 'b/b',
        'content_id': '012b/b',
        'parent_content_id': apex_node_b_content_id(),
        'title': 'taxon b - child b',
    }
    b_child_b_child_a_fixture = {
        'base_path': 'b/b/a',
        'content_id': '345b/b/a',
        'parent_content_id': '012b/b',
        'title': 'taxon b - child b - child a',
    }
    for taxon in [apex_node_a_fixture, apex_node_b_fixture, b_child_a_fixture, b_child_b_fixture, b_child_b_child_a_fixture, b_child_b_child_a_fixture]:
        node = Node(taxon, nodes)
        nodes[taxon['content_id']] = node
    return nodes

def test_base_path_must_be_present():
    node_data = {
        'base_path': '',
        'content_id': '123a',
        'title': 'taxon a',
    }
    with pytest.raises(ValueError) as node_error:
        Node(node_data, nodes)
    assert node_error.type is ValueError
    assert "base_path must be a string of length at least one" == node_error.value.args[0]

def test_title_must_be_present():
    node_data = {
        'base_path': 'a/',
        'content_id': '123a',
        'title': '',
    }
    with pytest.raises(ValueError) as node_error:
        Node(node_data, nodes)
    assert node_error.type is ValueError
    assert "title must be a string of length at least one" == node_error.value.args[0]

def test_content_id_must_be_present():
    node_data = {
        'base_path': 'a/',
        'content_id': '',
        'title': 'taxon a',
    }
    with pytest.raises(ValueError) as node_error:
        Node(node_data, nodes)
    assert node_error.type is ValueError
    assert "content_id must be a string of length at least one" == node_error.value.args[0]

def test_apex_node_is_apex_node(apex_node_with_children_fixture):
    assert apex_node_with_children_fixture.is_apex()

def test_child_node_is_not_an_apex_node(leaf_node_fixture):
    assert leaf_node_fixture.is_apex() == False

def test_apex_node_recursive_children_includes_all_children(apex_node_with_children_fixture):
    assert len(apex_node_with_children_fixture.recursive_children()) == 5
    assert ['taxon b', 'taxon b - child a', 'taxon b - child b', 'taxon b - child b - child a', 'taxon b - child b - child a'] == [node.title for node in apex_node_with_children_fixture.recursive_children()]

def test_apex_node_with_no_recursive_children_is_just_itself(apex_node_with_no_children_fixture):
    assert apex_node_with_no_children_fixture.recursive_children() == [apex_node_with_no_children_fixture]

def test_leaf_node_recursive_children_is_just_itself(leaf_node_fixture):
    assert leaf_node_fixture.recursive_children() == [leaf_node_fixture]

def test_unique_title_is_the_title_and_content_id(apex_node_with_children_fixture):
    assert '456b taxon b' == apex_node_with_children_fixture.unique_title()

def test_recursive_parents_is_all_parents_for_leaf_node(leaf_node_fixture):
    assert len(leaf_node_fixture.recursive_parents()) == 3
    assert ['taxon b - child b - child a', 'taxon b - child b', 'taxon b'] == [node.title for node in leaf_node_fixture.recursive_parents()]

def test_title_and_parent_title_for_apex_node_is_just_the_taxon_name(apex_node_with_children_fixture):
    print(apex_node_with_children_fixture.title_and_parent_title())
    assert "taxon b" == apex_node_with_children_fixture.title_and_parent_title()

def test_title_and_parent_title_for_leaf_node_misses_out_intermediate_nodes(leaf_node_fixture):
    assert "taxon b - child b ... > ... taxon b - child b - child a" == leaf_node_fixture.title_and_parent_title()

def test_title_and_parent_title_for_node_with_one_parent(level_two_node_fixture):
    assert "taxon b ... > ... taxon b - child a" == level_two_node_fixture.title_and_parent_title()

def test_all_siblings_and_children_for_apex_node_with_children(apex_node_with_children_fixture):
    assert 4 == len(apex_node_with_children_fixture.all_siblings_and_children())
    assert ['taxon b - child a', 'taxon b - child b', 'taxon b - child b - child a', 'taxon b - child b - child a'] == [node.title for node in apex_node_with_children_fixture.all_siblings_and_children()]

def test_all_siblings_and_children_for_apex_node_with_no_children(apex_node_with_no_children_fixture):
    # Apex nodes are not considered siblings
    assert 0 == len(apex_node_with_no_children_fixture.all_siblings_and_children())
    assert [] == apex_node_with_no_children_fixture.all_siblings_and_children()

def test_all_siblings_and_children_for_level_two_node(level_two_node_fixture):
    assert 3 == len(level_two_node_fixture.all_siblings_and_children())
    assert ['taxon b - child b', 'taxon b - child b - child a', 'taxon b - child b - child a'] == [node.title for node in level_two_node_fixture.all_siblings_and_children()]

def test_all_siblings_and_children_for_leaf_node(leaf_node_fixture):
    assert 0 == len(leaf_node_fixture.all_siblings_and_children())
    assert [] == leaf_node_fixture.all_siblings_and_children()
