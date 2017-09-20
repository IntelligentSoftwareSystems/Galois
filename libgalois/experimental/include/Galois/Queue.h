/** Efficient priority queues -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#ifndef GALOIS_QUEUE_H
#define GALOIS_QUEUE_H

#include "Galois/optional.h"
#include "Galois/Substrate/PaddedLock.h"
#include "Galois/Runtime/Mem.h"
#include "Galois/Substrate/PerThreadStorage.h"

#include <boost/utility.hpp>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <vector>
#include <sys/time.h>

namespace galois {

template<typename T, typename K>
int JComp(T& c, const K& k1, const K& k2) {
  if (c(k1, k2)) return -1;
  if (c(k2, k1)) return 1;
  else return 0;
}

struct ConcurrentSkipListMapHelper {
  /**
   * Special value used to identify base-level header
   */
  static int BASE_HEADER;
};

/**
 * Lock-free priority queue based on ConcurrentSkipListMap from the
 * Java Concurrent Collections. Translated from Java to C++ by
 * Donald Nguyen. No attempt was made to add explicit garbage collection,
 * so this implementation leaks memory.
 *
 * It is likely that this implementation provides more functionality than you
 * need, and you will find better performance with a more specific data
 * structure.
 *
 * From:
 *  http://www.java2s.com/Code/Java/Collections-Data-Structure/ConcurrentSkipListMap.htm
 */
/*
 * Written by Doug Lea with assistance from members of JCP JSR-166
 * Expert Group and released to the public domain, as explained at
 * http://creativecommons.org/licenses/publicdomain
 */
/**
 * A scalable {@link ConcurrentNavigableMap} implementation. This class
 * maintains a map in ascending key order, sorted according to the <i>natural
 * order</i> for the key's class (see {@link Comparable}), or by the
 * {@link Comparator} provided at creation time, depending on which constructor
 * is used.
 * 
 * <p>
 * This class implements a concurrent variant of <a
 * href="http://www.cs.umd.edu/~pugh/">SkipLists</a> providing expected average
 * <i>log(n)</i> time cost for the <tt>containsKey</tt>, <tt>get</tt>,
 * <tt>put</tt> and <tt>remove</tt> operations and their variants.
 * Insertion, removal, update, and access operations safely execute concurrently
 * by multiple threads. Iterators are <i>weakly consistent</i>, returning
 * elements reflecting the state of the map at some point at or since the
 * creation of the iterator. They do <em>not</em> throw {@link
 * ConcurrentModificationException}, and may proceed concurrently with other
 * operations. Ascending key ordered views and their iterators are faster than
 * descending ones.
 * 
 * <p>
 * All <tt>Map.Entry</tt> pairs returned by methods in this class and its
 * views represent snapshots of mappings at the time they were produced. They do
 * <em>not</em> support the <tt>Entry.setValue</tt> method. (Note however
 * that it is possible to change mappings in the associated map using
 * <tt>put</tt>, <tt>putIfAbsent</tt>, or <tt>replace</tt>, depending
 * on exactly which effect you need.)
 * 
 * <p>
 * Beware that, unlike in most collections, the <tt>size</tt> method is
 * <em>not</em> a constant-time operation. Because of the asynchronous nature
 * of these maps, determining the current number of elements requires a
 * traversal of the elements. Additionally, the bulk operations <tt>putAll</tt>,
 * <tt>equals</tt>, and <tt>clear</tt> are <em>not</em> guaranteed to be
 * performed atomically. For example, an iterator operating concurrently with a
 * <tt>putAll</tt> operation might view only some of the added elements.
 * 
 * <p>
 * This class and its views and iterators implement all of the <em>optional</em>
 * methods of the {@link Map} and {@link Iterator} interfaces. Like most other
 * concurrent collections, this class does not permit the use of <tt>null</tt>
 * keys or values because some null return values cannot be reliably
 * distinguished from the absence of elements.
 * 
 * @author Doug Lea
 * @param <K>
 *          the type of keys maintained by this map
 * @param <V>
 *          the type of mapped values
 */
template<typename K, typename V, typename Compare = std::less<K> >
class ConcurrentSkipListMap : private boost::noncopyable {
  /*
   * This class implements a tree-like two-dimensionally linked skip list in
   * which the index levels are represented in separate nodes from the base
   * nodes holding data. There are two reasons for taking this approach instead
   * of the usual array-based structure: 1) Array based implementations seem to
   * encounter more complexity and overhead 2) We can use cheaper algorithms for
   * the heavily-traversed index lists than can be used for the base lists.
   * Here's a picture of some of the basics for a possible list with 2 levels of
   * index:
   * 
   * Head nodes Index nodes +-+ right +-+ +-+ |2|---------------->|
   * |--------------------->| |->null +-+ +-+ +-+ | down | | v v v +-+ +-+ +-+
   * +-+ +-+ +-+ |1|----------->| |->| |------>| |----------->| |------>|
   * |->null +-+ +-+ +-+ +-+ +-+ +-+ v | | | | | Nodes next v v v v v +-+ +-+
   * +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ |
   * |->|A|->|B|->|C|->|D|->|E|->|F|->|G|->|H|->|I|->|J|->|K|->null +-+ +-+ +-+
   * +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
   * 
   * The base lists use a variant of the HM linked ordered set algorithm. See
   * Tim Harris, "A pragmatic implementation of non-blocking linked lists"
   * http://www.cl.cam.ac.uk/~tlh20/publications.html and Maged Michael "High
   * Performance Dynamic Lock-Free Hash Tables and List-Based Sets"
   * http://www.research.ibm.com/people/m/michael/pubs.htm. The basic idea in
   * these lists is to mark the "next" pointers of deleted nodes when deleting
   * to avoid conflicts with concurrent insertions, and when traversing to keep
   * track of triples (predecessor, node, successor) in order to detect when and
   * how to unlink these deleted nodes.
   * 
   * Rather than using mark-bits to mark list deletions (which can be slow and
   * space-intensive using AtomicMarkedReference), nodes use direct CAS'able
   * next pointers. On deletion, instead of marking a pointer, they splice in
   * another node that can be thought of as standing for a marked pointer
   * (indicating this by using otherwise impossible field values). Using plain
   * nodes acts roughly like "boxed" implementations of marked pointers, but
   * uses new nodes only when nodes are deleted, not for every link. This
   * requires less space and supports faster traversal. Even if marked
   * references were better supported by JVMs, traversal using this technique
   * might still be faster because any search need only read ahead one more node
   * than otherwise required (to check for trailing marker) rather than
   * unmasking mark bits or whatever on each read.
   * 
   * This approach maintains the essential property needed in the HM algorithm
   * of changing the next-pointer of a deleted node so that any other CAS of it
   * will fail, but implements the idea by changing the pointer to point to a
   * different node, not by marking it. While it would be possible to further
   * squeeze space by defining marker nodes not to have key/value fields, it
   * isn't worth the extra type-testing overhead. The deletion markers are
   * rarely encountered during traversal and are normally quickly garbage
   * collected. (Note that this technique would not work well in systems without
   * garbage collection.)
   * 
   * In addition to using deletion markers, the lists also use nullness of value
   * fields to indicate deletion, in a style similar to typical lazy-deletion
   * schemes. If a node's value is null, then it is considered logically deleted
   * and ignored even though it is still reachable. This maintains proper
   * control of concurrent replace vs delete operations -- an attempted replace
   * must fail if a delete beat it by nulling field, and a delete must return
   * the last non-null value held in the field. (Note: Null, rather than some
   * special marker, is used for value fields here because it just so happens to
   * mesh with the Map API requirement that method get returns null if there is
   * no mapping, which allows nodes to remain concurrently readable even when
   * deleted. Using any other marker value here would be messy at best.)
   * 
   * Here's the sequence of events for a deletion of node n with predecessor b
   * and successor f, initially:
   * 
   * +------+ +------+ +------+ ... | b |------>| n |----->| f | ... +------+
   * +------+ +------+
   * 
   * 1. CAS n's value field from non-null to null. From this point on, no public
   * operations encountering the node consider this mapping to exist. However,
   * other ongoing insertions and deletions might still modify n's next pointer.
   * 
   * 2. CAS n's next pointer to point to a new marker node. From this point on,
   * no other nodes can be appended to n. which avoids deletion errors in
   * CAS-based linked lists.
   * 
   * +------+ +------+ +------+ +------+ ... | b |------>| n
   * |----->|marker|------>| f | ... +------+ +------+ +------+ +------+
   * 
   * 3. CAS b's next pointer over both n and its marker. From this point on, no
   * new traversals will encounter n, and it can eventually be GCed. +------+
   * +------+ ... | b |----------------------------------->| f | ... +------+
   * +------+
   * 
   * A failure at step 1 leads to simple retry due to a lost race with another
   * operation. Steps 2-3 can fail because some other thread noticed during a
   * traversal a node with null value and helped out by marking and/or
   * unlinking. This helping-out ensures that no thread can become stuck waiting
   * for progress of the deleting thread. The use of marker nodes slightly
   * complicates helping-out code because traversals must track consistent reads
   * of up to four nodes (b, n, marker, f), not just (b, n, f), although the
   * next field of a marker is immutable, and once a next field is CAS'ed to
   * point to a marker, it never again changes, so this requires less care.
   * 
   * Skip lists add indexing to this scheme, so that the base-level traversals
   * start close to the locations being found, inserted or deleted -- usually
   * base level traversals only traverse a few nodes. This doesn't change the
   * basic algorithm except for the need to make sure base traversals start at
   * predecessors (here, b) that are not (structurally) deleted, otherwise
   * retrying after processing the deletion.
   * 
   * Index levels are maintained as lists with volatile next fields, using CAS
   * to link and unlink. Races are allowed in index-list operations that can
   * (rarely) fail to link in a new index node or delete one. (We can't do this
   * of course for data nodes.) However, even when this happens, the index lists
   * remain sorted, so correctly serve as indices. This can impact performance,
   * but since skip lists are probabilistic anyway, the net result is that under
   * contention, the effective "p" value may be lower than its nominal value.
   * And race windows are kept small enough that in practice these failures are
   * rare, even under a lot of contention.
   * 
   * The fact that retries (for both base and index lists) are relatively cheap
   * due to indexing allows some minor simplifications of retry logic. Traversal
   * restarts are performed after most "helping-out" CASes. This isn't always
   * strictly necessary, but the implicit backoffs tend to help reduce other
   * downstream failed CAS's enough to outweigh restart cost. This worsens the
   * worst case, but seems to improve even highly contended cases.
   * 
   * Unlike most skip-list implementations, index insertion and deletion here
   * require a separate traversal pass occuring after the base-level action, to
   * add or remove index nodes. This adds to single-threaded overhead, but
   * improves contended multithreaded performance by narrowing interference
   * windows, and allows deletion to ensure that all index nodes will be made
   * unreachable upon return from a public remove operation, thus avoiding
   * unwanted garbage retention. This is more important here than in some other
   * data structures because we cannot null out node fields referencing user
   * keys since they might still be read by other ongoing traversals.
   * 
   * Indexing uses skip list parameters that maintain good search performance
   * while using sparser-than-usual indices: The hardwired parameters k=1, p=0.5
   * (see method randomLevel) mean that about one-quarter of the nodes have
   * indices. Of those that do, half have one level, a quarter have two, and so
   * on (see Pugh's Skip List Cookbook, sec 3.4). The expected total space
   * requirement for a map is slightly less than for the current implementation
   * of java.util.TreeMap.
   * 
   * Changing the level of the index (i.e, the height of the tree-like
   * structure) also uses CAS. The head index has initial level/height of one.
   * Creation of an index with height greater than the current level adds a
   * level to the head index by CAS'ing on a new top-most head. To maintain good
   * performance after a lot of removals, deletion methods heuristically try to
   * reduce the height if the topmost levels appear to be empty. This may
   * encounter races in which it possible (but rare) to reduce and "lose" a
   * level just as it is about to contain an index (that will then never be
   * encountered). This does no structural harm, and in practice appears to be a
   * better option than allowing unrestrained growth of levels.
   * 
   * The code for all this is more verbose than you'd like. Most operations
   * entail locating an element (or position to insert an element). The code to
   * do this can't be nicely factored out because subsequent uses require a
   * snapshot of predecessor and/or successor and/or value fields which can't be
   * returned all at once, at least not without creating yet another object to
   * hold them -- creating such little objects is an especially bad idea for
   * basic internal search operations because it adds to GC overhead. (This is
   * one of the few times I've wished Java had macros.) Instead, some traversal
   * code is interleaved within insertion and removal operations. The control
   * logic to handle all the retry conditions is sometimes twisty. Most search
   * is broken into 2 parts. findPredecessor() searches index nodes only,
   * returning a base-level predecessor of the key. findNode() finishes out the
   * base-level search. Even with this factoring, there is a fair amount of
   * near-duplication of code to handle variants.
   * 
   * For explanation of algorithms sharing at least a couple of features with
   * this one, see Mikhail Fomitchev's thesis
   * (http://www.cs.yorku.ca/~mikhail/), Keir Fraser's thesis
   * (http://www.cl.cam.ac.uk/users/kaf24/), and Hakan Sundell's thesis
   * (http://www.cs.chalmers.se/~phs/).
   * 
   * Given the use of tree-like index nodes, you might wonder why this doesn't
   * use some kind of search tree instead, which would support somewhat faster
   * search operations. The reason is that there are no known efficient
   * lock-free insertion and deletion algorithms for search trees. The
   * immutability of the "down" links of index nodes (as opposed to mutable
   * "left" fields in true trees) makes this tractable using only CAS
   * operations.
   * 
   * Notation guide for local variables Node: b, n, f for predecessor, node,
   * successor Index: q, r, d for index node, right, down. t for another index
   * node Head: h Levels: j Keys: k, key Values: v, value Comparisons: c
   */

  struct HeadIndex;
  struct SnapshotEntry;

  /**
   * The topmost head index of the skiplist.
   */
  std::atomic<HeadIndex*> head;

  /**
   * Seed for simple random number generator. Not volatile since it doesn't
   * matter too much if different threads don't see updates.
   */
  galois::substrate::PerThreadStorage<int> randomSeed;
  Compare comp;

  galois::runtime::FixedSizeHeap node_heap;
  galois::runtime::FixedSizeHeap index_heap;
  galois::runtime::FixedSizeHeap head_index_heap;

  /**
   * Initialize or reset state. Needed by constructors, clone, clear,
   * readObject. and ConcurrentSkipListSet.clone. (Note that comparator must be
   * separately initialized.)
   */
  void initialize() {
    for (int i = 0, iend = randomSeed.size(); i < iend; ++i) {
      struct timeval time;
      int c = gettimeofday(&time, NULL);
      assert(c == 0);
      c = 0; // suppress warning
      // Add slight jitter so threads will get different seeds
      // and ensure non-zero
      *randomSeed.getRemote(i) = ((1000000 + i) * time.tv_sec + time.tv_usec) | 0x0100;
    }

    Node *node = new (node_heap.allocate(sizeof(Node))) 
      Node(K(), &ConcurrentSkipListMapHelper::BASE_HEADER, NULL);
    head = new (head_index_heap.allocate(sizeof(HeadIndex)))
      HeadIndex(node, NULL, NULL, 1);
  }

  /**
   * compareAndSet head node
   */
  bool casHead(HeadIndex* cmp, HeadIndex* val) {
    //return __sync_bool_compare_and_swap((uintptr_t*)&head, reinterpret_cast<uintptr_t>(cmp), reinterpret_cast<uintptr_t>(val));
    return head.compare_exchange_strong(cmp, val);
  }

  /* ---------------- Nodes -------------- */

  /**
   * Nodes hold keys and values, and are singly linked in sorted order, possibly
   * with some intervening marker nodes. The list is headed by a dummy node
   * accessible as head.node. The value field is declared only as Object because
   * it takes special non-V values for marker and header nodes.
   */
  struct Node {
    galois::optional<K> key;
    bool voidKey;
    std::atomic<void*> value;
    std::atomic<Node*> next;

    /**
     * compareAndSet value field
     */
    bool casValue(void* cmp, void* val) {
      return value.compare_exchange_strong(cmp, val);
      //return __sync_bool_compare_and_swap((uintptr_t*)&value, reinterpret_cast<uintptr_t>(cmp), reinterpret_cast<uintptr_t>(val));
    }

    /**
     * compareAndSet next field
     */
    bool casNext(Node* cmp, Node* val) {
      //return __sync_bool_compare_and_swap((uintptr_t*)&next, reinterpret_cast<uintptr_t>(cmp), reinterpret_cast<uintptr_t>(val));
      return next.compare_exchange_strong(cmp, val);
    }

    /**
     * Return true if this node is the header of base-level list.
     * @return true if this node is header node
     */
    bool isBaseHeader() {
      return value == &ConcurrentSkipListMapHelper::BASE_HEADER;
    }
    
    /**
     * Tries to append a deletion marker to this node.
     * 
     * @param f
     *          the assumed current successor of this node
     * @return true if successful
     */
    bool appendMarker(Node* f) {
      // TODO(ddn): Cannot easily switch this allocate to FixedSizeAllocator (yet)
      return casNext(f, new Node(f));
    }

    /**
     * Helps out a deletion by appending marker or unlinking from predecessor.
     * This is called during traversals when value field seen to be null.
     * 
     * @param b
     *          predecessor
     * @param f
     *          successor
     */
    void helpDelete(Node* b, Node* f) {
      /*
       * Rechecking links and then doing only one of the help-out stages per
       * call tends to minimize CAS interference among helping threads.
       */
      if (f == next && this == b->next) {
        if (f == NULL || f->value != f) // not already marked
          appendMarker(f);
        else
          b->casNext(this, f->next);
      }
    }

    /**
     * Return value if this node contains a valid key-value pair, else null.
     * 
     * @return this node's value if it isn't a marker or header or is deleted,
     *         else null.
     */
    V* getValidValue() {
      V* v = value;
      if (v == this || v == &ConcurrentSkipListMapHelper::BASE_HEADER)
        return NULL;
      return v;
    }

    /**
     * Create and return a new SnapshotEntry holding current mapping if this
     * node holds a valid value, else null
     * 
     * @return new entry or null
     */
    SnapshotEntry createSnapshot() {
      V* v = getValidValue();
      if (v == NULL)
        return SnapshotEntry();
      return SnapshotEntry(*key, v);
    }
  public:
    /**
     * Creates a new regular node.
     */
    Node(const K& k, void* v, Node* n): key(k), voidKey(false), value(v), next(n)  { }

    /**
     * Creates a new marker node. A marker is distinguished by having its value
     * field point to itself. Marker nodes also have null keys, a fact that is
     * exploited in a few places, but this doesn't distinguish markers from the
     * base-level header node (head.node), which also has a null key.
     */
    Node(Node* n): voidKey(true), value(this), next(n) { }
  };

  /* ---------------- Indexing -------------- */

  /**
   * Index nodes represent the levels of the skip list. To improve search
   * performance, keys of the underlying nodes are cached. Note that even though
   * both Nodes and Indexes have forward-pointing fields, they have different
   * types and are handled in different ways, that can't nicely be captured by
   * placing field in a shared abstract class.
   */
  struct Index {
    const galois::optional<K> key;
    /*final*/Node* node;
    /*final*/ Index* down;
    std::atomic<Index*> right;

    /**
     * Creates index node with given values
     */
    Index(Node* n, Index* d, Index* r): key(n->key), node(n), down(d), right(r) { }

    /**
     * compareAndSet right field
     */
    bool casRight(Index* cmp, Index* val) {
      //return __sync_bool_compare_and_swap((uintptr_t*)&right, reinterpret_cast<uintptr_t>(cmp), reinterpret_cast<uintptr_t>(val)); 
      return right.compare_exchange_strong(cmp, val);
    }

    /**
     * Returns true if the node this indexes has been deleted.
     * 
     * @return true if indexed node is known to be deleted
     */
    bool indexesDeletedNode() {
      return node->value == NULL;
    }

    /**
     * Tries to CAS newSucc as successor. To minimize races with unlink that may
     * lose this index node, if the node being indexed is known to be deleted,
     * it doesn't try to link in.
     * 
     * @param succ
     *          the expected current successor
     * @param newSucc
     *          the new successor
     * @return true if successful
     */
    bool link(Index* succ, Index* newSucc) {
      Node* n = node;
      newSucc->right = succ;
      return n->value != NULL && casRight(succ, newSucc);
    }

    /**
     * Tries to CAS right field to skip over apparent successor succ. Fails
     * (forcing a retraversal by caller) if this node is known to be deleted.
     * 
     * @param succ
     *          the expected current successor
     * @return true if successful
     */
    bool unlink(Index* succ) {
      return !indexesDeletedNode() && casRight(succ, succ->right);
    }
  };

  /* ---------------- Head nodes -------------- */

  /**
   * Nodes heading each level keep track of their level.
   */
  struct HeadIndex : public Index {
    const int level;

    HeadIndex(Node* n, Index* d, Index* r, int l): Index(n, d, r), level(l) { }
  };

  /* ---------------- Map.Entry support -------------- */

  /**
   * An immutable representation of a key-value mapping as it existed at some
   * point in time. This class does <em>not</em> support the
   * <tt>Map.Entry.setValue</tt> method.
   */
  struct SnapshotEntry {
    galois::optional<K> key;
    V* value;
    const bool valid;

    /**
     * Creates a new entry representing the given key and value.
     * 
     * @param key
     *          the key
     * @param value
     *          the value
     */
    SnapshotEntry(const K& k, V* v): key(k), value(v), valid(true) { }
    SnapshotEntry(): valid(false) { }
  };

  /* ---------------- Traversal -------------- */

  /**
   * Return a base-level node with key strictly less than given key, or the
   * base-level header if there is no such node. Also unlinks indexes to deleted
   * nodes found along the way. Callers rely on this side-effect of clearing
   * indices to deleted nodes.
   * 
   * @param key
   *          the key
   * @return a predecessor of key
   */
  Node* findPredecessor(const K& key) {
    for (;;) {
      Index* q = head;
      for (;;) {
        Index *d, *r;
        if ((r = q->right) != NULL) {
          if (r->indexesDeletedNode()) {
            if (q->unlink(r))
              continue; // reread r
            else
              break; // restart
          }
          if (JComp(comp, key, *r->key) > 0) {
            q = r;
            continue;
          }
        }
        if ((d = q->down) != NULL)
          q = d;
        else
          return q->node;
      }
    }
  }

  /**
   * Return node holding key or null if no such, clearing out any deleted nodes
   * seen along the way. Repeatedly traverses at base-level looking for key
   * starting at predecessor returned from findPredecessor, processing
   * base-level deletions as encountered. Some callers rely on this side-effect
   * of clearing deleted nodes.
   * 
   * Restarts occur, at traversal step centered on node n, if:
   * 
   * (1) After reading n's next field, n is no longer assumed predecessor b's
   * current successor, which means that we don't have a consistent 3-node
   * snapshot and so cannot unlink any subsequent deleted nodes encountered.
   * 
   * (2) n's value field is null, indicating n is deleted, in which case we help
   * out an ongoing structural deletion before retrying. Even though there are
   * cases where such unlinking doesn't require restart, they aren't sorted out
   * here because doing so would not usually outweigh cost of restarting.
   * 
   * (3) n is a marker or n's predecessor's value field is null, indicating
   * (among other possibilities) that findPredecessor returned a deleted node.
   * We can't unlink the node because we don't know its predecessor, so rely on
   * another call to findPredecessor to notice and return some earlier
   * predecessor, which it will do. This check is only strictly needed at
   * beginning of loop, (and the b.value check isn't strictly needed at all) but
   * is done each iteration to help avoid contention with other threads by
   * callers that will fail to be able to change links, and so will retry
   * anyway.
   * 
   * The traversal loops in doPut, doRemove, and findNear all include the same
   * three kinds of checks. And specialized versions appear in doRemoveFirst,
   * doRemoveLast, findFirst, and findLast. They can't easily share code because
   * each uses the reads of fields held in locals occurring in the orders they
   * were performed.
   * 
   * @param key
   *          the key
   * @return node holding key, or null if no such.
   */
  Node* findNode(const K& key) {
    for (;;) {
      Node* b = findPredecessor(key);
      Node* n = b->next;
      for (;;) {
        if (n == NULL)
          return NULL;
        Node* f = n->next;
        if (n != b->next) // inconsistent read
          break;
        void* v = n->value;
        if (v == NULL) { // n is deleted
          n->helpDelete(b, f);
          break;
        }
        if (v == n || b->value == NULL) // b is deleted
          break;
	int c = JComp(comp, key, *n->key);
	if (c == 0)
          return n;
        else if (c < 0)
          return NULL;
        b = n;
        n = f;
      }
    }
  }

#if 0
  /**
   * Specialized variant of findNode to perform Map.get. Does a weak
   * traversal, not bothering to fix any deleted index nodes,
   * returning early if it happens to see key in index, and passing
   * over any deleted base nodes, falling back to getUsingFindNode
   * only if it would otherwise return value from an ongoing
   * deletion. Also uses "bound" to eliminate need for some
   * comparisons (see Pugh Cookbook). Also folds uses of null checks
   * and node-skipping because markers have null keys.
   * @param okey the key
   * @return the value, or null if absent
   */
  V* doGet(const K key) {
    Node* bound = 0;
    Index* q = head;
    Index* r = q->right;
    Node* n = 0;
    K k;
    int c;
    for (;;) {
      Index* d;
      // Traverse rights
      if (r != 0 && (n = r->node) != bound) {
	k = n->key;
	if (!n->voidKey) {
	  if ((c = JComp(comp, key,k) > 0)) {
	    q = r;
	    r = r->right;
	    continue;
	  } else if (c == 0) {
	    V* v = (V*)n->value;
	    return (v != 0) ? v : getUsingFindNode(key);
	  } else
	    bound = n;
	}
      }
      
      // Traverse down
      if ((d = q->down) != 0) {
	q = d;
	r = d->right;
      } else
	break;
    }
    
    // Traverse nexts
    for (n = q->node->next;  n != 0; n = n->next) {
      k = n->key;
      if (!n->voidKey) {
	if ((c = JComp(comp,key,k)) == 0) {
	  V* v = (V*)n->value;
	  return (v != 0) ? v : getUsingFindNode(key);
	} else if (c < 0)
	  break;
      }
    }
    return 0;
  }
  
  /**
   * Perform map.get via findNode.  Used as a backup if doGet
   * encounters an in-progress deletion.
   * @param key the key
   * @return the value, or null if absent
   */
  V* getUsingFindNode(K key) {
    /*
     * Loop needed here and elsewhere in case value field goes
     * null just as it is about to be returned, in which case we
     * lost a race with a deletion, so must retry.
     */
    for (;;) {
      Node* n = findNode(key);
      if (n == 0)
	return 0;
      V* v = (V*)n->value;
      if (v != 0)
	return v;
    }
  }
#endif
    /**
   * Specialized variant of findNode to perform Map.get. Does a weak traversal,
   * not bothering to fix any deleted index nodes, returning early if it happens
   * to see key in index, and passing over any deleted base nodes, falling back
   * to getUsingFindNode only if it would otherwise return value from an ongoing
   * deletion. Also uses "bound" to eliminate need for some comparisons (see
   * Pugh Cookbook). Also folds uses of null checks and node-skipping because
   * markers have null keys.
   * 
   * @param okey
   *          the key
   * @return the value, or null if absent
   */
  V* doGet(const K& key) {
    //return getUsingFindNode(key);
    Node* bound = NULL;
    Index* q = head;
    for (;;) {
      K rk;
      Index *d, *r;
      if ((r = q->right) && (rk = *r->key) && r->node != bound) {
        int c = JComp(comp, key, rk);
        if (c > 0) {
          q = r;
          continue;
        }
        if (c == 0) {
          V* v = static_cast<V*>(r->node->value.load());
          return v ? v : getUsingFindNode(key);
        }
        bound = r->node; 
      }
      if ((d = q->down))
        q = d;
      else {
        for (Node* n = q->node->next; n; n = n->next) {
          if (!n->voidKey) {
            int c = JComp(comp, key, *n->key);
            if (c == 0) {
              V* v = static_cast<V*>(n->value.load());
              return v ? v : getUsingFindNode(key);
            }
            if (c < 0)
              return NULL;
          }
        }
        return NULL;
      }
    }
  }

  /**
   * Perform map.get via findNode. Used as a backup if doGet encounters an
   * in-progress deletion.
   * 
   * @param key
   *          the key
   * @return the value, or null if absent
   */
  V* getUsingFindNode(const K& key) {
    /*
     * Loop needed here and elsewhere in case value field goes null just as it
     * is about to be returned, in which case we lost a race with a deletion, so
     * must retry.
     */
    for (;;) {
      Node* n = findNode(key);
      if (!n)
        return NULL;
      V* v = static_cast<V*>(n->value.load());
      if (v)
        return v;
    }
  }

  /* ---------------- Insertion -------------- */

  /**
   * Main insertion method. Adds element if not present, or replaces value if
   * present and onlyIfAbsent is false.
   * 
   * @param kkey
   *          the key
   * @param value
   *          the value that must be associated with key
   * @param onlyIfAbsent
   *          if should not insert if already present
   * @return the old value, or null if newly inserted
   */
  V* doPut(const K& kkey, V* value, bool onlyIfAbsent) {
    for (;;) {
      Node* b = findPredecessor(kkey);
      Node* n = b->next;
      for (;;) {
        if (n != NULL) {
          Node* f = n->next;
          if (n != b->next) // inconsistent read
            break;
          ;
          void* v = n->value;
          if (v == NULL) { // n is deleted
            n->helpDelete(b, f);
            break;
          }
          if (v == n || b->value == NULL) // b is deleted
            break;
          int c = JComp(comp, kkey, *n->key);
	  if (c > 0) {
	    b = n;
	    n = f;
	    continue;
	  }
	  if (c == 0) {
	    if (onlyIfAbsent || n->casValue(v, value))
	      return static_cast<V*>(v);
	    else
	      break; // restart if lost race to replace value
	  }
          // else c < 0; fall through
        }

        Node* z = new (node_heap.allocate(sizeof(Node))) Node(kkey, value, n);
        if (!b->casNext(n, z))
          break; // restart if lost race to append to b
        int level = randomLevel();
        if (level > 0)
          insertIndex(z, level);
        return NULL;
      }
    }
  }

  /**
   * Return a random level for inserting a new node. Hardwired to k=1, p=0.5,
   * max 31.
   * 
   * This uses a cheap pseudo-random function that according to
   * http://home1.gte.net/deleyd/random/random4.html was used in Turbo Pascal.
   * It seems the fastest usable one here. The low bits are apparently not very
   * random (the original used only upper 16 bits) so we traverse from highest
   * bit down (i.e., test sign), thus hardly ever use lower bits.
   */
  int randomLevel() {
    int level = 0;
    int& seed = *randomSeed.getLocal();
    int r = seed;
    seed = r * 134775813 + 1;
    if (r < 0) {
      while ((r <<= 1) > 0)
        ++level;
    }
    return level;
  }

  /**
   * Create and add index nodes for given node.
   * 
   * @param z
   *          the node
   * @param level
   *          the level of the index
   */
  void insertIndex(Node* z, int level) {
    HeadIndex* h = head;
    int max = h->level;

    if (level <= max) {
      Index* idx = NULL;
      for (int i = 1; i <= level; ++i)
        idx = new (index_heap.allocate(sizeof(Index))) Index(z, idx, NULL);
      addIndex(idx, h, level);

    } else { // Add a new level
      /*
       * To reduce interference by other threads checking for empty levels in
       * tryReduceLevel, new levels are added with initialized right pointers.
       * Which in turn requires keeping levels in an array to access them while
       * creating new head index nodes from the opposite direction.
       */
      level = max + 1;
      //Index** idxs = new Index*[level+1];
      std::vector<Index*> idxs;
      idxs.resize(level+1);
      Index* idx = NULL;
      for (int i = 1; i <= level; ++i)
        idxs[i] = idx = new (index_heap.allocate(sizeof(Index))) Index(z, idx, NULL);

      HeadIndex* oldh;
      int k;
      for (;;) {
        oldh = head;
        int oldLevel = oldh->level;
        if (level <= oldLevel) { // lost race to add level
          k = level;
          break;
        }
        HeadIndex* newh = oldh;
        Node* oldbase = oldh->node;
        for (int j = oldLevel + 1; j <= level; ++j)
          newh = new (head_index_heap.allocate(sizeof(HeadIndex))) 
            HeadIndex(oldbase, newh, idxs[j], j);
        if (casHead(oldh, newh)) {
          k = oldLevel;
          break;
        }
      }
      addIndex(idxs[k], oldh, k);
    }
  }

  /**
   * Add given index nodes from given level down to 1.
   * 
   * @param idx
   *          the topmost index node being inserted
   * @param h
   *          the value of head to use to insert. This must be snapshotted by
   *          callers to provide correct insertion level
   * @param indexLevel
   *          the level of the index
   */
  void addIndex(Index* idx, HeadIndex* h, int indexLevel) {
    // Track next level to insert in case of retries
    int insertionLevel = indexLevel;

    // Similar to findPredecessor, but adding index nodes along
    // path to key.
    for (;;) {
      Index* q = h;
      Index* t = idx;
      int j = h->level;
      for (;;) {
        Index* r = q->right;
        if (r != NULL) {
          // compare before deletion check avoids needing recheck
	  int c = JComp(comp, *idx->key, *r->key);
          if (r->indexesDeletedNode()) {
            if (q->unlink(r))
              continue;
            else
              break;
          }
          if (c > 0) {
            q = r;
            continue;
          }
        }

        if (j == insertionLevel) {
          // Don't insert index if node already deleted
          if (t->indexesDeletedNode()) {
            findNode(*idx->key); // cleans up
            return;
          }
          if (!q->link(r, t))
            break; // restart
          if (--insertionLevel == 0) {
            // need final deletion check before return
            if (t->indexesDeletedNode())
              findNode(*idx->key);
            return;
          }
        }

        if (j > insertionLevel && j <= indexLevel)
          t = t->down;
        q = q->down;
        --j;
      }
    }
  }

  /* ---------------- Deletion -------------- */

  /**
   * Possibly reduce head level if it has no nodes. This method can (rarely)
   * make mistakes, in which case levels can disappear even though they are
   * about to contain index nodes. This impacts performance, not correctness. To
   * minimize mistakes as well as to reduce hysteresis, the level is reduced by
   * one only if the topmost three levels look empty. Also, if the removed level
   * looks non-empty after CAS, we try to change it back quick before anyone
   * notices our mistake! (This trick works pretty well because this method will
   * practically never make mistakes unless current thread stalls immediately
   * before first CAS, in which case it is very unlikely to stall again
   * immediately afterwards, so will recover.)
   * 
   * We put up with all this rather than just let levels grow because otherwise,
   * even a small map that has undergone a large number of insertions and
   * removals will have a lot of levels, slowing down access more than would an
   * occasional unwanted reduction.
   */
  void tryReduceLevel() {
    HeadIndex* h = head;
    HeadIndex* d;
    HeadIndex* e;
    if (h->level > 3 && (d = static_cast<HeadIndex*>(h->down)) != NULL
        && (e = static_cast<HeadIndex*>(d->down)) != NULL 
        && e->right.load() == NULL && d->right.load() == NULL
        && h->right.load() == NULL && casHead(h, d) && // try to set
        h->right.load() != NULL) // recheck
      casHead(d, h); // try to backout
  }

  /* ---------------- Finding and removing first element -------------- */

  /**
   * Specialized variant of findNode to get first valid node
   * 
   * @return first node or null if empty
   */
  Node* findFirst() const {
    for (;;) {
      Node* b = head.load()->node;
      Node* n = b->next;
      if (n == NULL)
        return NULL;
      if (n->value != NULL)
        return n;
      n->helpDelete(b, n->next);
    }
  }

  /**
   * Remove first entry; return either its key or a snapshot.
   * 
   * @param keyOnly
   *          if true return key, else return SnapshotEntry (This is a little
   *          ugly, but avoids code duplication.)
   * @return null if empty, first key if keyOnly true, else key,value entry
   */
  SnapshotEntry doRemoveFirst() {
    for (;;) {
      Node* b = head.load()->node;
      Node* n = b->next;
      if (n == NULL)
        return SnapshotEntry();
      Node* f = n->next;
      if (n != b->next)
        continue;
      void* v = n->value;
      if (v == NULL) {
        n->helpDelete(b, f);
        continue;
      }
      if (!n->casValue(v, NULL))
        continue;
      if (!n->appendMarker(f) || !b->casNext(n, f))
        findFirst(); // retry
      clearIndexToFirst();
      return SnapshotEntry(*n->key, static_cast<V*>(v));
    }
  }

  /**
   * Clear out index nodes associated with deleted first entry. Needed by
   * doRemoveFirst
   */
  void clearIndexToFirst() {
    for (;;) {
      Index* q = head;
      for (;;) {
        Index* r = q->right;
        if (r != NULL && r->indexesDeletedNode() && !q->unlink(r))
          break;
        if ((q = q->down) == NULL) {
          if (head.load()->right.load() == NULL)
            tryReduceLevel();
          return;
        }
      }
    }
  }


  /* ---------------- Constructors -------------- */
public:
  /**
   * Constructs a new empty map, sorted according to the keys' natural order.
   */
  ConcurrentSkipListMap():
    node_heap(sizeof(Node)),
    index_heap(sizeof(Index)),
    head_index_heap(sizeof(HeadIndex)) {
    initialize();
  }

#if 0
  /**
   * Constructs a new map containing the same mappings as the given
   * <tt>SortedMap</tt>, sorted according to the same ordering.
   * 
   * @param m
   *          the sorted map whose mappings are to be placed in this map, and
   *          whose comparator is to be used to sort this map.
   * @throws NullPointerException
   *           if the specified sorted map is <tt>null</tt>.
   */
  public ConcurrentSkipListMap(SortedMap<K, ? extends V> m) {
    this.comparator = m.comparator();
    initialize();
    buildFromSorted(m);
  }

  /**
   * Streamlined bulk insertion to initialize from elements of given sorted map.
   * Call only from constructor or clone method.
   */
  private void buildFromSorted(SortedMap<K, ? extends V> map) {
    if (map == null)
      throw new NullPointerException();

    HeadIndex<K, V> h = head;
    Node<K, V> basepred = h.node;

    // Track the current rightmost node at each level. Uses an
    // ArrayList to avoid committing to initial or maximum level.
    ArrayList<Index<K, V>> preds = new ArrayList<Index<K, V>>();

    // initialize
    for (int i = 0; i <= h.level; ++i)
      preds.add(null);
    Index<K, V> q = h;
    for (int i = h.level; i > 0; --i) {
      preds.set(i, q);
      q = q.down;
    }

    Iterator<? extends Map.Entry<? extends K, ? extends V>> it = map.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<? extends K, ? extends V> e = it.next();
      int j = randomLevel();
      if (j > h.level)
        j = h.level + 1;
      K k = e.getKey();
      V v = e.getValue();
      if (k == null || v == null)
        throw new NullPointerException();
      Node<K, V> z = new Node<K, V>(k, v, null);
      basepred.next = z;
      basepred = z;
      if (j > 0) {
        Index<K, V> idx = null;
        for (int i = 1; i <= j; ++i) {
          idx = new Index<K, V>(z, idx, null);
          if (i > h.level)
            h = new HeadIndex<K, V>(h.node, h, idx, i);

          if (i < preds.size()) {
            preds.get(i).right = idx;
            preds.set(i, idx);
          } else
            preds.add(idx);
        }
      }
    }
    head = h;
  }
#endif

  /* ------ Map API methods ------ */

  /**
   * Returns the value to which this map maps the specified key.  Returns
   * <tt>null</tt> if the map contains no mapping for this key.
   *
   * @param key key whose associated value is to be returned.
   * @return the value to which this map maps the specified key, or
   *               <tt>null</tt> if the map contains no mapping for the key.
   * @throws ClassCastException if the key cannot be compared with the keys
   *                  currently in the map.
   * @throws NullPointerException if the key is <tt>null</tt>.
   */
  V* get(const K& key) {
    return doGet(key);
    //return getUsingFindNode(key);
  }


  /**
   * Associates the specified value with the specified key in this map. If the
   * map previously contained a mapping for this key, the old value is replaced.
   * 
   * @param key
   *          key with which the specified value is to be associated.
   * @param value
   *          value to be associated with the specified key.
   * 
   * @return previous value associated with specified key, or <tt>null</tt> if
   *         there was no mapping for key.
   * @throws ClassCastException
   *           if the key cannot be compared with the keys currently in the map.
   * @throws NullPointerException
   *           if the key or value are <tt>null</tt>.
   */
  V* put(const K& key, V* value) {
    assert(value != NULL);
    return doPut(key, value, false);
  }

  /**
   * Returns the number of elements in this map. If this map contains more than
   * <tt>Integer.MAX_VALUE</tt> elements, it returns
   * <tt>Integer.MAX_VALUE</tt>.
   * 
   * <p>
   * Beware that, unlike in most collections, this method is <em>NOT</em> a
   * constant-time operation. Because of the asynchronous nature of these maps,
   * determining the current number of elements requires traversing them all to
   * count them. Additionally, it is possible for the size to change during
   * execution of this method, in which case the returned result will be
   * inaccurate. Thus, this method is typically not very useful in concurrent
   * applications.
   * 
   * @return the number of elements in this map.
   */
  int size() {
    long count = 0;
    for (Node n = findFirst(); n != NULL; n = n->next) {
      if (n->getValidValue() != NULL)
        ++count;
    }
    return (count >= std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max() : (int) count;
  }

  /**
   * Returns <tt>true</tt> if this map contains no key-value mappings.
   * 
   * @return <tt>true</tt> if this map contains no key-value mappings.
   */
  bool isEmpty() const {
    return findFirst() == NULL;
  }

  /* ------ ConcurrentMap API methods ------ */

  /**
   * If the specified key is not already associated with a value, associate it
   * with the given value. This is equivalent to
   * 
   * <pre>
   * if (!map.containsKey(key))
   *   return map.put(key, value);
   * else
   *   return map.get(key);
   * </pre>
   * 
   * except that the action is performed atomically.
   * 
   * @param key
   *          key with which the specified value is to be associated.
   * @param value
   *          value to be associated with the specified key.
   * @return previous value associated with specified key, or <tt>null</tt> if
   *         there was no mapping for key.
   * 
   * @throws ClassCastException
   *           if the key cannot be compared with the keys currently in the map.
   * @throws NullPointerException
   *           if the key or value are <tt>null</tt>.
   */
  V* putIfAbsent(const K& key, V* value) {
    assert(value != NULL);
    return doPut(key, value, true);
  }

  /**
   * Returns a key-value mapping associated with the least key in this map, or
   * <tt>null</tt> if the map is empty. The returned entry does <em>not</em>
   * support the <tt>Entry.setValue</tt> method.
   * 
   * @return an Entry with least key, or <tt>null</tt> if the map is empty.
   */
  SnapshotEntry firstEntry() {
    for (;;) {
      Node* n = findFirst();
      if (n == NULL)
        return SnapshotEntry();
      SnapshotEntry e = n->createSnapshot();
      if (e.valid)
        return e;
    }
  }

  /**
   * Returns a key-value mapping associated with the greatest
   * key in this map, or <tt>null</tt> if the map is empty.
   * The returned entry does <em>not</em> support
   * the <tt>Entry.setValue</tt> method.
   *
   * @return an Entry with greatest key, or <tt>null</tt>
   * if the map is empty.
   */
  std::pair<K, V*> lastEntry() {
    for (;;) {
      Node* n = findLast();
      if (n == 0)
	return std::make_pair((K)0,(V*)NULL);
      SnapshotEntry e = n->createSnapshot();
      if (e.valid)
	return std::make_pair((K)e.key, (V*)e.value);
    }
  }

  /**
   * Removes and returns a key-value mapping associated with the least key in
   * this map, or <tt>null</tt> if the map is empty. The returned entry does
   * <em>not</em> support the <tt>Entry.setValue</tt> method.
   * 
   * @return the removed first entry of this map, or <tt>null</tt> if the map
   *         is empty.
   */
  galois::optional<V*> pollFirstValue() {
    SnapshotEntry retval = doRemoveFirst();
    if (retval.valid)
      return galois::optional<V*>(retval.value);
    else
      return galois::optional<V*>();
  }

  /**
   * Remove first entry; return key or null if empty.
   */
  galois::optional<K> pollFirstKey() {
    SnapshotEntry retval = doRemoveFirst();
    if (retval.valid)
      return retval.key;
    else
      return galois::optional<K>();
  }

  /* ---------------- Finding and removing last element -------------- */

  /**
   * Specialized version of find to get last valid node
   * @return last node or null if empty
   */
  Node* findLast() {
    /*
     * findPredecessor can't be used to traverse index level
     * because this doesn't use comparisons.  So traversals of
     * both levels are folded together.
     */
    Index* q = head;
    for (;;) {
      Index* d;
      Index* r;
      if ((r = q->right) != 0) {
	if (r->indexesDeletedNode()) {
	  q->unlink(r);
	  q = head; // restart
	}
	else
	  q = r;
      } else if ((d = q->down) != 0) {
	q = d;
      } else {
	Node* b = q->node;
	Node* n = b->next;
	for (;;) {
	  if (n == 0)
	    return b->isBaseHeader() ? 0 : b;
	  Node* f = n->next;            // inconsistent read
	  if (n != b->next)
	    break;
	  void* v = n->value;
	  if (v == 0) {                 // n is deleted
	    n->helpDelete(b, f);
	    break;
	  }
	  if (v == n || b->value == 0)   // b is deleted
	    break;
	  b = n;
	  n = f;
	}
	q = head; // restart
      }
    }
  }

};

template<class T, class Compare=std::less<T>, class Alloc=std::allocator<T> >
class PairingHeap: private boost::noncopyable {
  /*
   * Conceptually a pairing heap is an n-ary tree. We represent this using
   * a fixed number of fields:
   *  left: the node's first child
   *  prev, next: a doubly linked list of a node's siblings, where the first
   *    child's prev points to the parent, i.e., x->left->prev == x.
   */
  struct Node {
    T value;
    Node* left;
    Node* next;
    Node* prev;
    Node(T v) : value(v), left(NULL), next(NULL), prev(NULL) { }
  };

public:
  typedef typename Alloc::template rebind<Node>::other allocator_type;

private:
  Compare comp;
  const allocator_type& alloc;
  std::vector<Node*, typename allocator_type::template rebind<Node*>::other> m_tree;
  Node* m_root;
  int m_capacity;

  allocator_type get_alloc() {
    return allocator_type(alloc);
  }

  /**
   * Merge two trees together, preserving heap property. Make the least tree the
   * parent of the other. Return new root.
   */
  Node* merge(Node* a, Node* b) {
    assert(!a->next);
    assert(!b->next);
    assert(!a->prev);
    assert(!b->prev);

    Node *first = a, *second = b;
    if (comp(b->value, a->value)) {
      first = b;
      second = a;
    }

    second->prev = first;
    second->next = first->left;
    if (first->left)
      first->left->prev = second;
    first->left = second;

    assert(!first->prev);
    assert(!first->next);
    return first;
  }

  /**
   * Heapify children, return new root.
   */
  Node* mergeChildren(Node* root) {
    if (!root->left)
      return NULL;

    // Breakup kids
    for (Node* kid = root->left; kid; kid = kid->next) {
      m_tree.push_back(kid);
      kid->prev->next = NULL;
      kid->prev = NULL;
    }

    Node* retval;
    if (m_tree.size() == 1) {
      retval = m_tree[0];
    } else {
      // Merge in pairs
      unsigned i = 0;
      for (; i + 1 < m_tree.size(); i += 2) {
        m_tree[i] = merge(m_tree[i], m_tree[i+1]);
      }

      // Merge pairs together
      if (i != m_tree.size()) {
        // odd number of siblings and m_tree[i] is the last
        m_tree[i-2] = merge(m_tree[i-2], m_tree[i]);
      }
      for (unsigned j = (m_tree.size() >> 1) - 1; j > 0; --j) {
        m_tree[2*(j-1)] = merge(m_tree[2*(j-1)], m_tree[2*j]);
      }

      retval = m_tree[0];
    }
    m_tree.clear();
    return retval;
  }

  /**
   * Remove and deallocate root node and return new root.
   */
  Node* deleteMin(Node* root) {
    Node* retval = mergeChildren(root);
    get_alloc().destroy(root);
    get_alloc().deallocate(root, 1);
    return retval;
  }

  /**
   * Remove node from heap, putting node in it's own heap.
   */
  void detach(Node* node) {
    if (node == m_root)
      return;

    if (node->prev->left == node) {
      // node is someone's left-most child
      node->prev->left = node->next;
    } else {
      node->prev->next = node->next;
    }
    if (node->next)
      node->next->prev = node->prev;
    node->next = NULL;
    node->prev = NULL;
  }

public:
  typedef Node* Handle;

  PairingHeap(const allocator_type& a = allocator_type()):
    alloc(a), m_tree(a), m_root(NULL) { }

  ~PairingHeap() {
    while (!empty()) {
      pollMin();
    }
  }

  bool empty() const {
    return m_root == NULL;
  }

  Handle add(const T& x) {
    typename allocator_type::pointer node = get_alloc().allocate(1);
    get_alloc().construct(node, Node(x));

    if (!m_root)
      m_root = node;
    else
      m_root = merge(m_root, node);

    return node;
  }

  T value(Handle node) {
    return node->value;
  }

  void decreaseKey(Handle node, const T& val) {
    assert(comp(val, node->value));

    node->value = val;
    if (node != m_root) {
      detach(node);
      m_root = merge(m_root, node);
    }
  }

  void deleteNode(Handle node) {
    if (node == m_root) {
      pollMin();
    } else {
      detach(node);
      node = deleteMin(node);
      if (node)
        m_root = merge(m_root, node);
    }
  }

  galois::optional<T> pollMin() {
    if (empty())
      return galois::optional<T>();
    T retval = m_root->value;
    m_root = deleteMin(m_root);

    return galois::optional<T>(retval);
  }
};

template<class T,class Compare=std::less<T>,bool Concurrent=true>
class FCPairingHeap: private boost::noncopyable {
  struct Op {
    T item;
    galois::optional<T> retval;
    Op* response;
    bool req;
  };

  struct Slot {
    std::atomic<Op*> req __attribute__((aligned(64)));
    std::atomic<Slot*> next;
    std::atomic<Slot*> prev;
    Slot(): req(NULL), next(NULL), prev(NULL) { }
  };

  galois::substrate::PerThreadStorage<Slot*> localSlots;
  galois::substrate::PerThreadStorage<std::vector<Op*> > ops;
  galois::substrate::PaddedLock<Concurrent> lock;
  PairingHeap<T,Compare> heap;
  std::atomic<Slot*> slots;
  const int maxTries;

  void flatCombine() {
    for (int tries = 0; tries < maxTries; ++tries) {
      //_GLIBCXX_READ_MEM_BARRIER;
      int changes = 0;
      Slot* cur = slots.load(std::memory_order_acquire);
      while (cur->next) {
        Op* op = cur->req;
        if (op && op->req) {
          ++changes;
          if (op->response) {
            // pollMin op
            op->response->retval = heap.pollMin();
          } else {
            // add op
            heap.add(op->item);
          }
          cur->req = op->response;
        } 

        cur = cur->next;
      }
      if (changes)
        break;
    }
  }

  void addSlot(Slot* slot) {
    Slot* cur;
    do {
      cur = slots;
      slot->next.store(cur, std::memory_order_relaxed);
    //} while (!__sync_bool_compare_and_swap((uintptr_t*)&slots, reinterpret_cast<uintptr_t>(cur), reinterpret_cast<uintptr_t>(slot)));
    } while (!slots.compare_exchange_strong(cur, slot));
    cur->prev.store(slot, std::memory_order_release);
  }

  Slot* getMySlot() {
    Slot*& mySlot = *localSlots.getLocal();
    if (mySlot == NULL) {
      mySlot = new Slot();
      addSlot(mySlot);
    }

    return mySlot;
  }

  Op* getOp() {
    std::vector<Op*>& myOps = *ops.getLocal();
    if (myOps.empty()) {
      return new Op;
    }
    Op* retval = myOps.back();
    myOps.pop_back();
    return retval;
  }

  void recycleOp(Op* op) {
    ops.getLocal()->push_back(op);
  }

  Op* makeAddReq(const T& value) {
    Op* req = getOp();
    req->item = value;
    req->response = NULL;
    req->req = true;
    return req;
  }

  Op* makePollReq() {
    Op* response = getOp();
    response->req = false;
    response->response = NULL;

    Op* req = getOp();
    req->req = true;
    req->response = response;

    return req;
  }

public:
  FCPairingHeap(): maxTries(1) { 
    slots = new Slot();
  }
  
  ~FCPairingHeap() {
    Slot* cur = slots;
    while (cur) {
      Slot* t = cur->next;
      delete cur;
      cur = t;
    }

    for (unsigned int i = 0; i < ops.size(); ++i) {
      std::vector<Op*>& v = *ops.getRemote(i);
      for (typename std::vector<Op*>::iterator ii = v.begin(), ee = v.end(); ii != ee ; ++ii) {
        delete *ii;
      }
    }
  }

  void add(T value) {
    Slot* mySlot = getMySlot();
    //Slot* volatile& myNext = mySlot->next;
    std::atomic<Op*>& myReq = mySlot->req;
    Op* req = makeAddReq(value);
    myReq.store(req, std::memory_order_release);

    do {
      //if (!myNext) {
      //  addSlot(mySlot);
      //  assert(0 && "should never happen");
      //  abort();
      //}

      if (lock.try_lock()) {
        flatCombine();
        lock.unlock();
        recycleOp(req);
        return;
      } else {
        //_GLIBCXX_WRITE_MEM_BARRIER;
        while (myReq.load(std::memory_order_acquire) == req) {
	  galois::substrate::asmPause();
        }
        //_GLIBCXX_READ_MEM_BARRIER;
        recycleOp(req);
        return;
      }
    } while(1);
  }

  galois::optional<T> pollMin() {
    Slot* mySlot = getMySlot();
    //Slot* volatile& myNext = mySlot->next;
    std::atomic<Op*>& myReq = mySlot->req;
    Op* req = makePollReq();
    myReq.store(req, std::memory_order_release);

    do {
      //if (!myNext) {
      //  addSlot(mySlot);
      //  assert(0 && "should never happen");
      //  abort();
      //}

      if (lock.try_lock()) {
        flatCombine();
        lock.unlock();

	galois::optional<T> retval = myReq.load(std::memory_order_relaxed)->retval;
        recycleOp(req->response);
        recycleOp(req);
        return retval;
      } else {
        //_GLIBCXX_WRITE_MEM_BARRIER;
        while (myReq.load(std::memory_order_acquire) == req) {
	  galois::substrate::asmPause();
        }
        //_GLIBCXX_READ_MEM_BARRIER;
	galois::optional<T> retval = myReq.load(std::memory_order_acquire)->retval;
        recycleOp(req->response);
        recycleOp(req);
        return retval;
      }
    } while(1);
  }

  bool empty() const {
    lock.lock();
    bool retval = heap.empty();
    lock.unlock();
    return retval;
  }

};

}
#endif
