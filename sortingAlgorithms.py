import sys
import os
import random
import time
from typing import List, Tuple, Any
import math
import copy

class SortingAlgorithms:
    def __init__(self, data: List = []):
        self.data = data
        self.comparisons = 0
        self.swaps = 0

    def bubble_sort(self, arr: List) -> List:
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
        return arr

    def quick_sort(self, arr: List, low: int, high: int) -> None:
        if low < high:
            pi = self._partition(arr, low, high)
            self.quick_sort(arr, low, pi - 1)
            self.quick_sort(arr, pi + 1, high)

    def _partition(self, arr: List, low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            self.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.swaps += 1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swaps += 1
        return i + 1

    def merge_sort(self, arr: List) -> List:
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            self.merge_sort(left)
            self.merge_sort(right)
            i = j = k = 0
            while i < len(left) and j < len(right):
                self.comparisons += 1
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1
            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1
        return arr

    def heap_sort(self, arr: List) -> List:
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.swaps += 1
            self._heapify(arr, i, 0)
        return arr

    def _heapify(self, arr: List, n: int, i: int) -> None:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            self.comparisons += 1
            if arr[left] > arr[largest]:
                largest = left
        if right < n:
            self.comparisons += 1
            if arr[right] > arr[largest]:
                largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.swaps += 1
            self._heapify(arr, n, largest)

    def insertion_sort(self, arr: List) -> List:
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                self.comparisons += 1
                arr[j + 1] = arr[j]
                j -= 1
                self.swaps += 1
            arr[j + 1] = key
        return arr

    def selection_sort(self, arr: List) -> List:
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            self.swaps += 1
        return arr

    def shell_sort(self, arr: List) -> List:
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    self.comparisons += 1
                    arr[j] = arr[j - gap]
                    j -= gap
                    self.swaps += 1
                arr[j] = temp
            gap //= 2
        return arr

    def counting_sort(self, arr: List) -> List:
        if not arr:
            return arr
        max_val = max(arr)
        min_val = min(arr)
        range_of_elements = max_val - min_val + 1
        count = [0] * range_of_elements
        output = [0] * len(arr)
        for i in range(len(arr)):
            count[arr[i] - min_val] += 1
        for i in range(1, len(count)):
            count[i] += count[i - 1]
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
        return output

    def radix_sort(self, arr: List) -> List:
        if not arr:
            return arr
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            self._counting_sort_for_radix(arr, exp)
            exp *= 10
        return arr

    def _counting_sort_for_radix(self, arr: List, exp: int) -> None:
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = (arr[i] // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]

    def bucket_sort(self, arr: List) -> List:
        if not arr:
            return arr
        bucket_count = 10
        max_val = max(arr)
        min_val = min(arr)
        buckets = [[] for _ in range(bucket_count)]
        for num in arr:
            index = int((num - min_val) * bucket_count /
                        (max_val - min_val + 1))
            buckets[index].append(num)
        for bucket in buckets:
            bucket.sort()
        result = []
        for bucket in buckets:
            result.extend(bucket)
        return result

    def cocktail_sort(self, arr: List) -> List:
        n = len(arr)
        swapped = True
        start = 0
        end = n - 1
        while swapped:
            swapped = False
            for i in range(start, end):
                self.comparisons += 1
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
                    self.swaps += 1
            if not swapped:
                break
            swapped = False
            end -= 1
            for i in range(end - 1, start - 1, -1):
                self.comparisons += 1
                if arr[i] > arr[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
                    self.swaps += 1
            start += 1
        return arr

    def comb_sort(self, arr: List) -> List:
        n = len(arr)
        gap = n
        shrink = 1.3
        sorted_flag = False
        while not sorted_flag:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted_flag = True
            i = 0
            while i + gap < n:
                self.comparisons += 1
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    self.swaps += 1
                    sorted_flag = False
                i += 1
        return arr

    def gnome_sort(self, arr: List) -> List:
        index = 0
        while index < len(arr):
            if index == 0:
                index += 1
            else:
                self.comparisons += 1
                if arr[index] >= arr[index - 1]:
                    index += 1
                else:
                    arr[index], arr[index - 1] = arr[index - 1], arr[index]
                    self.swaps += 1
                    index -= 1
        return arr

    def reset_stats(self) -> None:
        self.comparisons = 0
        self.swaps = 0

    def get_stats(self) -> Tuple[int, int]:
        return (self.comparisons, self.swaps)


class SortingBenchmark:
    def __init__(self):
        self.results = {}

    def generate_random_array(self, size: int, min_val: int = 0,
                              max_val: int = 1000) -> List[int]:
        return [random.randint(min_val, max_val) for _ in range(size)]

    def benchmark_algorithm(self, algo_name: str, sort_func,
                            arr: List) -> dict:
        test_arr = copy.deepcopy(arr)
        sorter = SortingAlgorithms(test_arr)
        start_time = time.time()
        if algo_name == "quick_sort":
            sort_func(sorter, test_arr, 0, len(test_arr) - 1)
        else:
            result = sort_func(sorter, test_arr)
        end_time = time.time()
        comparisons, swaps = sorter.get_stats()
        return {
            "time": end_time - start_time,
            "comparisons": comparisons,
            "swaps": swaps
        }

    def run_benchmarks(self, array_size: int = 100) -> None:
        test_array = self.generate_random_array(array_size)
        sorter = SortingAlgorithms()
        algorithms = {
            "bubble_sort": sorter.bubble_sort,
            "quick_sort": sorter.quick_sort,
            "merge_sort": sorter.merge_sort,
            "heap_sort": sorter.heap_sort,
            "insertion_sort": sorter.insertion_sort,
            "selection_sort": sorter.selection_sort,
            "shell_sort": sorter.shell_sort,
            "cocktail_sort": sorter.cocktail_sort,
            "comb_sort": sorter.comb_sort,
            "gnome_sort": sorter.gnome_sort
        }
        print(f"Running benchmarks on array of size {array_size}...")
        for name, func in algorithms.items():
            print(f"Testing {name}...")
            result = self.benchmark_algorithm(name, func, test_array)
            self.results[name] = result
            stat_str = f"{name}: Time={result['time']:.6f}s, "
            stat_str += f"Comparisons={result['comparisons']}, "
            stat_str += f"Swaps={result['swaps']}"
            print(stat_str)


def main():
    print("=== Sorting Algorithms Implementation ===")
    test_data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50, 23, 67, 89, 100, 1]
    print(f"Original array: {test_data}")
    sorter = SortingAlgorithms()
    bubble_result = sorter.bubble_sort(copy.deepcopy(test_data))
    print(f"Bubble Sort: {bubble_result}")
    sorter.reset_stats()
    merge_result = sorter.merge_sort(copy.deepcopy(test_data))
    print(f"Merge Sort: {merge_result}")
    sorter.reset_stats()
    quick_data = copy.deepcopy(test_data)
    sorter.quick_sort(quick_data, 0, len(quick_data) - 1)
    print(f"Quick Sort: {quick_data}")
    benchmark = SortingBenchmark()
    benchmark.run_benchmarks(50)


if __name__ == "__main__":
    main()
