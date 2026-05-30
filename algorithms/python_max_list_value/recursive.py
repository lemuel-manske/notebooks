number = list[int or float]

def max_arr_val(arr: list) -> number:
	
	if len(arr) == 0:
		return None
	
	if len(arr) == 2:
		return arr[0] if arr[0] > arr[1] else arr[1]

	sub_max = max_arr_val(arr[1:])

	return arr[0] if arr[0] > sub_max else sub_max

assert(max_arr_val([5,89,34,6,44,23,34])) == 89
assert(max_arr_val([1,225,23])) == 225