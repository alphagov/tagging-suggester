class Node:
    def __init__(self, entry, all_nodes):
        self.base_path = entry['base_path']
        self.content_id = entry['content_id']
        self.title = entry['title']
        if 'parent_content_id' in entry:
            self.parent = all_nodes[entry['parent_content_id']]
            self.parent.children.append(self)
        else:
            self.parent = None
        self.children = []
        self.all_sibs_and_children = None
    def unique_title(self):
        # Some taxa have identical names so using the title as the
        # key of a dictionary will cause problems
        return self.content_id[:3] + " " + self.title
    def recursive_parents(self):
        results = []
        if self.parent:
            results.append(self.parent.recursive_parents())
        else:
            results.append([self])
        # Set to make them unique
        flattened = self.__flatten(results)
        unique = list(set(flattened))
        return unique;
    def title_and_parent_title(self):
        if self.parent is not None:
            return self.parent.title + " ... > ... " + self.title
        else:
            return self.title;
    def recursive_children(self):
        results = []
        results.append([self])
        for child in self.children:
            results.append(child.recursive_children())
        # Set to make them unique
        flattened = self.__flatten(results)
        unique = list(set(flattened))
        return unique;
    def all_siblings_and_children(self):
        if self.all_sibs_and_children is None:
            results = []
            # This is a slightly hacky way of not returning the entire tree if the node
            # is a level 1 taxon. E.g. if the taxon has no parent it's level 1 so only return it's children
            # rather than all it's siblings and their children (which would be the entire tree)
            if not self.parent:
                results.append(self.recursive_children())
            else:
                for node in self.parent.children:
                    results.append(node.recursive_children())
            flattened_results = self.__flatten(results)
            # Remove self from results
            self.all_sibs_and_children = [result for result in flattened_results if result.content_id != self.content_id]
            return self.all_sibs_and_children
        else:
            return self.all_sibs_and_children
    def is_apex(self):
        return self.parent is None
    def __flatten(self, S):
        if S == []:
            return S
        if isinstance(S[0], list):
            return self.__flatten(S[0]) + self.__flatten(S[1:])
        return S[:1] + self.__flatten(S[1:])