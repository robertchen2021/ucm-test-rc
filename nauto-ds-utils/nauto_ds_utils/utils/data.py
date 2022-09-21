from typing import List


def find_closest_index(arr: List[float], target: float, comp=max) -> float:
    '''
    Modified Binary Search to find the nearest (`comp`) value (target)
    This was used instead of the find_closest index.
    '''
    if target <= arr[0]:
        return 0

    if target >= arr[len(arr) - 1]:
        return len(arr) - 1

    i, j = 0, len(arr)
    while i < j:
        mid = (i + j) // 2
        if arr[mid] == target:
            return mid
        if (target < arr[mid]):
            if (mid > 0 and target > arr[mid - 1]):
                return comp(mid - 1, mid)
            j = mid
        else:
            if (mid < len(arr) - 1 and target < arr[mid + 1]):
                return comp(mid, mid + 1)
            i = mid + 1
    return mid, arr[mid]    
    
