from typing import Union

number = list[int or float]

def arr_sum(arr: list) -> number:
	tot = 0

	for x in arr:
		tot += x
	
	return tot

assert(arr_sum([1,3,4,2,5])) == 15