from typing import Union

number = list[int or float]

def arr_sum(arr: list) -> number:
	if len(arr) == 0: # list == []
		return 0
	return arr[0] + arr_sum(arr[1:])

assert(arr_sum([1,2,3,4,5])) == 15

print(arr_sum([1,2,3,4,5]))