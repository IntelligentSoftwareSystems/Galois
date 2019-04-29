#ifndef BLISS_ORBIT_HH
#define BLISS_ORBIT_HH

namespace bliss {
class Orbit {
	class OrbitEntry {
		public:
			unsigned int element;
			OrbitEntry *next;
			unsigned int size;
	};
	OrbitEntry *orbits;
	OrbitEntry **in_orbit;
	unsigned int nof_elements;
	unsigned int _nof_orbits;
	void merge_orbits(OrbitEntry *orbit1, OrbitEntry *orbit2) {
		if(orbit1 != orbit2) {
			_nof_orbits--;
			// Only update the elements in the smaller orbit
			if(orbit1->size > orbit2->size) {
				OrbitEntry * const temp = orbit2;
				orbit2 = orbit1;
				orbit1 = temp;
			}
			// Link the elements of orbit1 to the almost beginning of orbit2
			OrbitEntry *e = orbit1;
			while(e->next) {
				in_orbit[e->element] = orbit2;
				e = e->next;
			}
			in_orbit[e->element] = orbit2;
			e->next = orbit2->next;
			orbit2->next = orbit1;
			// Keep the minimal orbit representative in the beginning
			if(orbit1->element < orbit2->element) {
				const unsigned int temp = orbit1->element;
				orbit1->element = orbit2->element;
				orbit2->element = temp;
			}
			orbit2->size += orbit1->size;
		}
	}

	public:
	// Create a new orbit information object.
	// The init() function must be called next to actually initialize the object.
	Orbit() {
		orbits = 0;
		in_orbit = 0;
		nof_elements = 0;
	}
	~Orbit() {
		if(orbits) {
			free(orbits);
			orbits = 0;
		}
		if(in_orbit) {
			free(in_orbit);
			in_orbit = 0;
		}
		nof_elements = 0;
	}

	// Initialize the orbit information to consider sets of \a N elements.
	// It is required that \a N > 0.
	// The orbit information is reset so that each element forms an orbit of its own.
	// Time complexity is O(N). \sa reset()
	void init(const unsigned int n) {
		assert(n > 0);
		if(orbits) free(orbits);
		orbits = (OrbitEntry*)malloc(n * sizeof(OrbitEntry));
		if(in_orbit) free(in_orbit);
		in_orbit = (OrbitEntry**)malloc(n * sizeof(OrbitEntry*));
		nof_elements = n;
		reset();
	}

	// Reset the orbits so that each element forms an orbit of its own.
	// Time complexity is O(N).
	void reset() {
		assert(orbits);
		assert(in_orbit);
		for(unsigned int i = 0; i < nof_elements; i++) {
			orbits[i].element = i;
			orbits[i].next = 0;
			orbits[i].size = 1;
			in_orbit[i] = &orbits[i];
		}
		_nof_orbits = nof_elements;
	}

	// Merge the orbits of the elements \a e1 and \a e2.
	// Time complexity is O(k), where k is the number of elements in
	// the smaller of the merged orbits.
	void merge_orbits(unsigned int e1, unsigned int e2) {
		merge_orbits(in_orbit[e1], in_orbit[e2]);
	}

	// Is the element \a e the smallest element in its orbit?
	// Time complexity is O(1).
	bool is_minimal_representative(unsigned element) const {
		return(get_minimal_representative(element) == element);
	}
	/// Get the smallest element in the orbit of the element \a e.
	// Time complexity is O(1).
	unsigned get_minimal_representative(unsigned element) const {
		OrbitEntry * const orbit = in_orbit[element];
		return(orbit->element);
	}
	// Get the number of elements in the orbit of the element \a e.
	// Time complexity is O(1).

	unsigned orbit_size(unsigned element) const {
		return(in_orbit[element]->size);
	}
	// Get the number of orbits.
	// Time complexity is O(1).
	unsigned int nof_orbits() const {return _nof_orbits; }
};

} // namespace bliss

#endif
