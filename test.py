import topics_frequent.array as topics
import pytest

# @pytest.fixture
# def array_with_dup():
#     return [0,0,1,1,1,2,2,3,3,4]

def test_26():
    nums = [0,0,1,1,1,2,2,3,3,4]
    print("Test question 26")
    assert topics.remove_duplicates_from_sorted_array_26().removeDuplicates(nums) == 5
    
def test_27():
    nums = [0,1,2,2,3,0,4,2]
    print("Test question 27")
    assert topics.remove_element_27().removeElement(nums, 2) == 5