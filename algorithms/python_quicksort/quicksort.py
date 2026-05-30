def quicksort(arr: list):
	if len(arr) < 2:
		return arr

	else:
		pivot = arr[0]
		less_than_pivot = [i for i in arr[1:] if i <= pivot]
		greater_than_pivot = [i for i in arr[1:] if i > pivot]

		return quicksort(less_than_pivot) + [pivot] + quicksort(greater_than_pivot)

assert(quicksort([10, 5, 2, 3])) == [2,3,5,10]

'''
1- [5, 2, 3] + [10] + []

2- [2, 3] + [5] + [10] + []

3- [2] + [3] + [5] + [10] + []
'''