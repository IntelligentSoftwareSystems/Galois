import random
import optparse
import collections
import sys

def main(num_users, num_movies, num_edges, options):
	random.seed(1000)
	num_nodes = num_users + num_movies
	adj = collections.defaultdict(set)
	
	#print('p sp %d %d' % (num_nodes, num_edges))

	user_set = set(xrange(1, num_users+1))

	def randUser():
		x = random.randint(num_movies+1, num_movies + num_users)
		return x	
	def randMovie():
		x =random.randint(1, num_movies)
		return x	
	def randRating():
		return random.randint(1, 5)
	def addEdge(src, dst, w):
		if dst in adj[src]:
			return False
		print('a %d %d %d' % (src, dst, w))
		adj[src].add(dst)
		return True
	
	edges_emitted = num_movies
	for movie in xrange(1, num_movies+1):
		user = randUser()
		addEdge(movie, user, randRating())
		user_set.discard(user)
	
	edges_emitted = edges_emitted + len(user_set)
	for user in user_set:
		while not addEdge(randMovie(), num_movies + user, randRating()):
			pass

	for i in xrange(num_edges - edges_emitted):
		while not addEdge(randMovie(), randUser(), randRating()):
			pass

if __name__ == '__main__':
	usage = 'usage: %prog <num users> <num movies> <num edges>'
	parser = optparse.OptionParser(usage=usage)
	(options, args) = parser.parse_args()
	if len(args) != 3:
		parser.error('missing arguments')
	main(int(args[0]), int(args[1]), int(args[2]), options)
