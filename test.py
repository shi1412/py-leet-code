import topics_frequent.array as topics
import pytest

# @pytest.fixture
# def array_with_dup():
#     return [0,0,1,1,1,2,2,3,3,4]

def test_11():
    height = [4,3,2,1,4]
    print("Test question 11")
    assert topics.container_with_most_water_11().maxArea(height) == 16

def test_26():
    nums = [0,0,1,1,1,2,2,3,3,4]
    print("Test question 26")
    assert topics.remove_duplicates_from_sorted_array_26().removeDuplicates(nums) == 5

def test_34():
     nums = [5,7,7,8,8,10]
     target = 6
     print("Test question 34")
     assert topics.find_first_and_last_postion_of_element_in_sorted_array_34().searchRange(nums, target)

def test_36():
     board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
     print("Test question 36")
     assert topics.valid_sudoku_36().isValidSudoku(board) == True

def test_37():
    board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
    print("Test question 37")
    assert topics.sudoku_solver_37().solveSudoku(board) == [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
    
def test_27():
    nums = [0,1,2,2,3,0,4,2]
    print("Test question 27")
    assert topics.remove_element_27().removeElement(nums, 2) == 5

def test_42():
    height = [4,2,0,3,2,5]
    print("Test question 42")
    assert topics.trapping_rain_water_42().trap(height) == 9

def test_48():
    matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    print("Test question 48")
    assert topics.rotate_image_48().rotate(matrix) == [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
    
def test_53():
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    print("Test question 53")
    assert topics.maximum_subarray_53().maxSubArray_A(nums) == 6
    assert topics.maximum_subarray_53().maxSubArray_B(nums) == 6

def test_54():
    matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    print("Test question 54")
    assert topics.spiral_matrix_54().spiralOrder(matrix) == [1,2,3,4,8,12,11,10,9,5,6,7]
    
def test_56():
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print("Test question 56")
    assert topics.merge_intervals_56().merge(intervals) == [[1,6],[8,10],[15,18]]

def test_59():
    print("Test question 59")
    assert topics.spiral_matrix_II_59().generateMatrix(3) == [[1,2,3],[8,9,4],[7,6,5]]
    assert topics.spiral_matrix_II_59().generateMatrix(1) == [[1]]

def test_73():
    matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    print("Test question 73")
    assert topics.set_matrix_zeroes_73().setZeroes(matrix) == [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

def test_134():
    gas = [1,2,3,4,5]
    cost = [3,4,5,1,2]
    print("Test question 134")
    assert topics.gas_station_134().canCompleteCircuit(gas, cost) == 3
    
def test_152():
    nums = [2,3,-2,4]
    print("Test question 152")
    assert topics.maximum_product_subarray_152().maxProduct(nums) == 6
    
def test_163():
    nums, l, u = [0,1,3,50,75], 0, 99
    nums1, l1, u1 = [], 1, 1
    nums2, l2, u2 = [], -3, -1
    print("Test question 163")
    assert topics.missing_range_163().findMissingRanges(nums, l, u) == ["2","4->49","51->74","76->99"]
    assert topics.missing_range_163().findMissingRanges(nums1, l1, u1) == ["1"]
    assert topics.missing_range_163().findMissingRanges(nums2, l2, u2) == ["-3->-1"]

def test_169():
    nums = [2,2,1,1,1,2,2]
    print("Test question 169")
    assert topics.majority_element_169().majorityElement(nums) == 2

def test_189():
    nums = [-1,-100,3,99]
    k = 2
    assert topics.rotate_array_189().rotate(nums, k) == [3,99,-1,-100]

def test_209():
    s = 7
    nums = [2,3,1,2,4,3]
    print("Test questions 209")
    assert topics.minimum_size_subarray_sum_209().minSubArray(s, nums) == 2 
    
def test_238():
    nums = [1, 2, 3, 4]
    print("Test question 238")
    assert topics.product_of_array_except_self_238().productExceptSelf(nums) == [24, 12, 8, 6]
    
def test_253():
    intervals = [[0,30],[5,10],[15,20]]
    print("Test question 253")
    assert topics.meeting_rooms_253().minMeetingRooms_A(intervals) == 2
    assert topics.meeting_rooms_253().minMeetingRooms_B(intervals) == 2

def test_274():
    citations = [3,0,6,1,5]
    print("Test question 274")
    assert topics.h_index_274().hIndex_A(citations) == 3
    
def test_280():
    nums =  [3,5,2,1,6,4]
    print("Test questions 280")
    assert topics.wiggle_sort_280().wiggleSort(nums) == [3,5,1,6,2,4]
    
def test_289():
    board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    print("Test question 289")
    assert topics.game_of_life_289().gameOfLife_A(board) == [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

def test_296():
    grid = [[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]]
    print("Test question 296")
    assert topics.best_meeting_point_296().minTotalDistance(grid) == 6

def test_311():
    A = [[1,0,0],[-1,0,3]]
    B = [[7,0,0],[0,0,0],[0,0,1]]
    print("Test question 311")
    assert topics.sparse_matrix_multiplicatoin_311().multiply(A, B) == [[7,0,0],[-7,0,3]]
    
def test_325():
    nums = [-2, -1, 2, 1]
    k = 1
    print("Test quesion 325")
    assert topics.maximum_size_subarray_sum_equals_k_325().maxSubArrayLen(nums, k) == 2
    
def test_370():
    length = 5
    updates = [[1,3,2],[2,4,3],[0,2,-2]]
    print("Test question 370")
    assert topics.range_addition_370().getModifiedArray(length, updates) == [-2,0,3,5,3]
    
def test_376():
    nums = [1,7,4,9,2,5]
    print("Test question 376")
    assert topics.wiggle_subsequence_376().wiggleMaxLength(nums) == 6