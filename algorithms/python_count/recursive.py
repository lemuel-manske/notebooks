def count(arr: list) -> int:
	if arr == []:
		return 0
	return 1 + count(arr[1:])

print(count([1,2,3,4,5,6,7,8,9,10])) # 10