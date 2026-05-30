grafo = {
	'inicio': {
		'a': 6,
		'b': 2
	},
	'a': {
		'fim': 1
	},
	'b': {
		'a': 3,
		'fim': 5
	},
	'fim': {}
}

infinito = float('inf')

custos = {
	'a': 6,
	'b': 4,
	'fim': infinito
}

pais = {
	'a': 'inicio',
	'b': 'inicio',
	'fim': None
}

processados = []

def achar_menor(custos):
	custo_mais_baixo = float('inf')
	nodo_custo_mais_baixo = None
	for nodo in custos:
		custo = custos[nodo]
		if custo < custo_mais_baixo and nodo not in processados:
			custo_mais_baixo = custo
			nodo_custo_mais_baixo = nodo
	return nodo_custo_mais_baixo

nodo = achar_menor(custos)

while nodo is not None:
	custo = custos[nodo]	
	vizinhos = grafo[nodo]

	for n in vizinhos.keys():
		novo_custo = custo + vizinhos[n]
		
		if custos[n] > novo_custo:
			custos[n] = novo_custo
			pais[n] = nodo
	processados.append(nodo)
	nodo = achar_menor(custos)

print(custos)
print(pais)