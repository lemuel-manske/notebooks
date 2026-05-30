# step by step:
# divide by 2
# the '?' is forward or backward
# cut the half by another half

array = [
	'Sonny',
	'Samuel',
	'Alisa',
	'Aaron',
	'Santos',
	'Baron',
	'Nathaniel',
	'Deacon',
	'Clinton',
	'Zain',
	'Ashanti',
	'Abbey',
	'Danielle',
	'Johnathan',
	'Valerie',
	'Tessa',
	'Lesly',
	'Asher',
	'Pedro',
	'Maren',
	'Bobby',
	'Brenden',
	'Elliot',
	'Benjamin',
	'Miracle',
	'Isiah',
	'Ricardo',
]

search = 'Asher'

def binary_search(list, item):
	low = 0
	high = len(list) - 1

	while low <= high:
		mid = (low + high) // 2
		guess = list[mid]

		if guess == item:
			return mid
		if guess > item:
			high = mid - 1
		else:
			low = mid + 1

	return None
	
names = sorted(array)
guess = 'Maren'

print(binary_search(names, guess))