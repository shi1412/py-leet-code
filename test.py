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
    
def test_56():
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print("Test question 56")
    assert topics.merge_intervals_56().merge(intervals) == [[1,6],[8,10],[15,18]]
    
def test_163():
    nums, l, u = [0,1,3,50,75], 0, 99
    nums1, l1, u1 = [], 1, 1
    nums2, l2, u2 = [], -3, -1
    print("Test question 163")
    assert topics.missing_range_163().findMissingRanges(nums, l, u) == ["2","4->49","51->74","76->99"]
    assert topics.missing_range_163().findMissingRanges(nums1, l1, u1) == ["1"]
    assert topics.missing_range_163().findMissingRanges(nums2, l2, u2) == ["-3->-1"]
    

def test_253():
    intervals = [[0,30],[5,10],[15,20]]
    print("Test question 253")
    assert topics.meeting_rooms_253().minMeetingRooms_A(intervals) == 2
    assert topics.meeting_rooms_253().minMeetingRooms_B(intervals) == 2