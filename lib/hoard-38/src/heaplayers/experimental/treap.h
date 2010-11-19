/* -*- C++ -*- */

#ifndef _TREAP_H_
#define _TREAP_H_

#include <assert.h>
#include <stdlib.h>
#include <limits.h>


// A Cartesian tree.
// adapted from treap code by Bobby Blumofe


template <class KEY, class VALUE>
class Treap {

public:

  class Node {   // A node in the treap.
    friend class Treap;
    unsigned int priority; //   The priority.
    KEY key;      //   The key.
    VALUE value;  //   The value.
    Node* parent; //   Pointer to parent.
    Node* left;   //   Pointer to left child.
    Node* right;  //   Pointer to right child.

  public:
    // Construct node.
    Node (void) : left (NULL), right (NULL) {}

    Node (unsigned int priority_, KEY key_, VALUE value_, Node* parent_)
      : priority (priority_), key (key_), value (value_),
	parent (parent_), left (NULL), right (NULL) {}
    KEY getKey (void) const { return key; }
    VALUE getValue (void) const { return value; }
  };

  // Construct an empty treap.
  Treap (void);

  // Destructor.
  ~Treap (void);

  // Return value of key or 0 if not found.
  // Return a matching node (or NULL if not found).
  Node * lookup (KEY key) const {
    return lookup_ (key);
  }

  // Return a matching node (or NULL if not found).
  Node * lookupGreater (KEY key) const {
    return lookupGreater_ (key);
  }

  // Set the given key to have the given value.
  void insert (Node * n, KEY key, VALUE value, unsigned int priority);

  // Remove entry with given key.
  // Remove entry.
  Node * remove (Node * node) {
#if 0
    // Search for node with given key.
    Node* node = lookup_ (key);
#endif
    
    // If not found, then do nothing.
    if (!node)
      return NULL;
    
    // While node is not a leaf...
    while (node->left || node->right) {
      
      // If left child only, rotate right.
      if (!node->right)
	rotateRight (node);
      
      // If right child only, rotate left.
      else if (!node->left)
	rotateLeft (node);
      
      // If both children,
      else {
	if (node->left->priority < node->right->priority)
	  rotateRight (node);
	else
	  rotateLeft (node);
      }
    }
    
    // Clip off node.
    Node* parent = node->parent;
    if (!parent) {
      assert (root == node);
      root = 0;
    }
    else {
      if (parent->left == node)
	parent->left = 0;
      else
	parent->right = 0;
    }
    
    // Check treap properties.
    // assert (heapProperty (root, INT_MIN));
    // assert (bstProperty (root, INT_MIN, INT_MAX));
    
#if 0
    delete node;
    return NULL;
#else
    // Return the removed node.
    
    return node;
#endif
  }


  void print (void) const { reallyPrint (root); cout << endl; }

  void reallyPrint (Node * node) const {
    if (node == NULL) return;
    reallyPrint (node->left);
    cout << "[" << node->key << "] ";
    reallyPrint (node->right);
  }



private:

  Node* root;  // Pointer to root node of treap.

  // Disable copy and assignment.
  Treap (const Treap& treap) {}
  Treap& operator= (const Treap& treap) { return *this; }

  // Check treap properties.
  int heapProperty (Node* node, int lbound) const;
  int bstProperty (Node* node, int lbound, int ubound) const;

  // Delete treap rooted at given node.
  void deleteTreap (Node* node);

  // Return node with given key or NULL if not found.
  Node* lookup_ (KEY key) const;
  Node* lookupGreater_ (KEY key) const;
  Node* lookupGeq (KEY key, Node * root) const;

  // Perform rotations.
  void rotateLeft (Node* node);
  void rotateRight (Node* node);
};


// Construct an empty treap.
template <class KEY, class VALUE>
Treap<KEY,VALUE>::Treap (void)
  : root (0)
{
}

// Destructor.
template <class KEY, class VALUE>
Treap<KEY,VALUE>::~Treap (void)
{
  deleteTreap (root);
}

// Delete treap rooted at given node.
template <class KEY, class VALUE>
void Treap<KEY,VALUE>::deleteTreap (Node* node)
{
  // If empty, nothing to do.
  if (!node)
    return;

  // Delete left and right subtreaps.
  deleteTreap (node->left);
  deleteTreap (node->right);

  // Delete root node.
  delete node;
}

// Test heap property in subtreap rooted at node.
template <class KEY, class VALUE>
int Treap<KEY,VALUE>::heapProperty (Node* node, int lbound) const
{
  // Empty treap satisfies.
  if (!node)
    return 1;

  // Check priority.
  if (node->priority < lbound)
    return 0;

  // Check left subtreap.
  if (!heapProperty (node->left, node->priority))
    return 0;

  // Check right subtreap.
  if (!heapProperty (node->right, node->priority))
    return 0;

  // All tests passed.
  return 1;
}

// Test bst property in subtreap rooted at node.
template <class KEY, class VALUE>
int Treap<KEY,VALUE>::bstProperty (Node* node, int lbound, int ubound) const
{
  // Empty treap satisfies.
  if (!node)
    return 1;

  // Check key in range.
  if (node->key < lbound || node->key > ubound)
    return 0;

  // Check left subtreap.
  if (!bstProperty (node->left, lbound, node->key))
    return 0;

  // Check right subtreap.
  if (!bstProperty (node->right, node->key, ubound))
    return 0;

  // All tests passed.
  return 1;
}

// Perform a left rotation.
template <class KEY, class VALUE>
void Treap<KEY,VALUE>::rotateLeft (Node* node)
{
  // Get right child.
  Node* right = node->right;
  assert (right);

  // Give node right's left child.
  node->right = right->left;

  // Adjust parent pointers.
  if (right->left)
    right->left->parent = node;
  right->parent = node->parent;

  // If node is root, change root.
  if (!node->parent) {
    assert (root == node);
    root = right;
  }

  // Link node parent to right.
  else {
    if (node->parent->left == node)
      node->parent->left = right;
    else
      node->parent->right = right;
  }

  // Put node to left of right.
  right->left = node;
  node->parent = right;
}

// Perform a right rotation.
template <class KEY, class VALUE>
void Treap<KEY,VALUE>::rotateRight (Node* node)
{
  // Get left child.
  Node* left = node->left;
  assert (left);

  // Give node left's right child.
  node->left = left->right;

  // Adjust parent pointers.
  if (left->right)
    left->right->parent = node;
  left->parent = node->parent;

  // If node is root, change root.
  if (!node->parent) {
    assert (root == node);
    root = left;
  }

  // Link node parent to left.
  else {
    if (node->parent->left == node)
      node->parent->left = left;
    else
      node->parent->right = left;
  }

  // Put node to right of left.
  left->right = node;
  node->parent = left;
}

// Return node with given key or 0 if not found.
template <class KEY, class VALUE>
Treap<KEY,VALUE>::Node* Treap<KEY,VALUE>::lookup_ (KEY key) const
{
  // Start at the root.
  Node* node = root;

  // While subtreap rooted at node not empty...
  while (node) {

    // If found, then return value.
    if (key == node->key)
      return node;

    // Otherwise, search left or right subtreap.
    else if (key < node->key)
      node = node->left;
    else
      node = node->right;
  }

  // Return.
  return node;
}


template <class KEY, class VALUE>
Treap<KEY,VALUE>::Node* Treap<KEY,VALUE>::lookupGreater_ (KEY key) const
{
  return lookupGeq (key, root);
}


// Return node with greater or equal key or 0 if not found.
template <class KEY, class VALUE>
Treap<KEY,VALUE>::Node* Treap<KEY,VALUE>::lookupGeq (KEY key, Node * rt) const
{
  Node * bestSoFar = NULL;

  // Start at the root.
  Node* node = rt;

  // While subtreap rooted at node not empty...
  while (node) {

    // If exact match found, then return value.
    if (key == node->key)
      return node;

    // Move right -- this node is too small.
    if (node->key < key)
      node = node->right;
    
    // Otherwise, this one's pretty good;
    // look for a better match.
    else {
      if ((bestSoFar == NULL) || (bestSoFar->key > node->key))
	bestSoFar = node;
      node = node->left;
    }
  }

  // Return.
  return bestSoFar;
}


// Set the given key to have the given value.
template <class KEY, class VALUE>
void Treap<KEY,VALUE>::insert (Treap<KEY,VALUE>::Node * n, KEY key, VALUE value, unsigned int priority)
{

  //  print();

  // 0 is not a valid value.
  assert (value != 0);

  // Start at the root.
  Node* parent = 0;
  Node* node = root;


  // While subtreap rooted at node not empty...
  while (node) {

#if 0
    // If found, then update value and done.
    if (key == node->key) {
      node->value = value;
      return;
    }
#endif

    // Otherwise, search left or right subtreap.
    parent = node;


    if (key < node->key)
      node = node->left;
    else
      node = node->right;
  }


  // Not found, so create new node.
  // EDB was
  // node = new Node (lrand48(), key, value, parent);
  node = new (n) Node (priority, key, value, parent);
  // node = new Node (priority, key, value, parent);

  // If the treap was empty, then new node is root.
  if (!parent)
    root = node;

  // Otherwise, add node as left or right child.
  else if (key < parent->key)
    parent->left = node;
  else
    parent->right = node;

  // While heap property not satisfied...
  while (parent && parent->priority > node->priority) {

    // Perform rotation.
    if (parent->left == node)
      rotateRight (parent);
    else
      rotateLeft (parent);

    // Move up.
    parent = node->parent;
  }

  // print();

  // Check treap properties.
  // assert (heapProperty (root, INT_MIN));
  // assert (bstProperty (root, INT_MIN, INT_MAX));
}



#endif // _TREAP_H_
