# 49 Group Anagrams
# dictionary
# string sorting
def groupAnagrams(strs):
        group = {}
        if len(strs) < 1:
            return strs
        
        for i in range(len(strs)):
            elem = strs[i]
            elem_sort = "".join(sorted(elem))
            if elem_sort in group:
                group[elem_sort].append(elem)
            else:
                group[elem_sort] = [elem]
                
        return group.values()  