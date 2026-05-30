def is_palindrom(str:str) -> bool:
	return f"{str} : {str == str[::-1]}"

def solve(str:str):
	for i in range(0, len(str)):
		if len(str[i:i+3]) > 2:
			print(is_palindrom(str[i:i+3]))

solve