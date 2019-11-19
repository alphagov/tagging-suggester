class Node:
    def __init__(self, entry, all_nodes):
        self.base_path = entry['base_path']
        if not self.base_path:
            raise ValueError("base_path must be a string of length at least one")

        self.content_id = entry['content_id']
        if not self.content_id:
            raise ValueError("content_id must be a string of length at least one")

        self.title = entry['title']
        if not self.title:
            raise ValueError("title must be a string of length at least one")

        if 'parent_content_id' in entry:
            self.parent = all_nodes[entry['parent_content_id']]
            self.parent.children.append(self)
        else:
            self.parent = None
        self.children = []
        self.all_sibs_and_children = None

    def unique_title(self):
        """
        Some taxa have identical names so prepend part of the content_id
        :return: String, title of taxon that is (almost guaranteed) to be unique
        """
        return self.content_id + " " + self.title

    def recursive_parents(self):
        """
        All parents of a taxon
        :return: List of all unique parents of a taxon
        """
        results = [self]
        if self.parent:
            results.append(self.parent.recursive_parents())
        # Set to make them unique
        flattened = self.__flatten(results)
        return self.__unique(flattened)

    def title_and_parent_title(self):
        """
        :return: String, title of taxon and it's parent, suitable for displaying to users
        """
        if self.parent is not None:
            return self.parent.title + " ... > ... " + self.title
        else:
            return self.title;

    def recursive_children(self):
        """
        :return: List, Node instances of all children and sub-children of a taxon
        """
        results = []
        results.append([self])
        for child in self.children:
            results.append(child.recursive_children())
        # Set to make them unique
        flattened = self.__flatten(results)
        return self.__unique(flattened)

    def all_siblings_and_children(self):
        """
        All siblings and children of a taxon, not including itself
        :return: List, Node instances of all siblings and children
        """
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
        """
        :return: Bool, whether a taxon is a top level taxon
        """
        return self.parent is None

    def __flatten(self, S):
        """
        Recursivey flattens a list of lists
        :return: List, flattened list elements in sub-lists
        """
        if S == []:
            return S
        if isinstance(S[0], list):
            return self.__flatten(S[0]) + self.__flatten(S[1:])
        return S[:1] + self.__flatten(S[1:])

    def __unique(self, list):
        """
        Ensure every item in a list is unique
        :return: List, where every item is unique in a consistent order
        """
        unique = []
        for item in list:
            if item not in unique:
                unique.append(item)
        return unique
