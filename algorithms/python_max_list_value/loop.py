number = list[int or float]

def max_arr_val(arr: list) -> number:
	
	max = arr[0]

	for x in arr:
		if x > max:
			max = x

	return max

assert(max_arr_val([1,2,3,4,5])) == 5