from collections import deque

grafo = {}

grafo['voce'] = ['alice', 'bob', 'claire']
grafo['bob'] = ['anuj', 'peggy']
grafo['alice'] = ['peggy']
grafo['claire'] = ['thom', 'jonny']
grafo['anuj'] = []
grafo['peggy'] = []
grafo['thom'] = []
grafo['jonny'] = []

def pessoa_e_vendedor(pessoa: str):
	return pessoa[-1] == 'm'

def search(name: str):
	fila = deque()
	fila += grafo[name]
	verificadas = []
	while fila:
		pessoa = fila.popleft()
		if not pessoa in verificadas:
			if pessoa_e_vendedor(pessoa):
				print(f'{pessoa} é um vendedor de manga!')
				return True
			else:
				fila += grafo[pessoa]
				verificadas.append(pessoa)
	return False
			
search('voce')