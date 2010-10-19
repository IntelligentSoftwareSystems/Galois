// Insert Bag implementation -*- C++ -*-

namespace threadsafe {
  
  template< class T>
  class ts_insert_bag {
    struct bag_item {
      T item;
      bag_item* next;

      explicit bag_item(const T& V)
	:item(V), next(0)
      {}
    };
    
    volatile bag_item* head;

    void i_push(bag_item* b) {
      volatile bag_item* H;
      do {
	H = head;
	b->next = const_cast<bag_item*>(H);
      } while (!__sync_bool_compare_and_swap(&head, H, b));
    }

  public:
    typedef T        value_type;
    typedef const T& const_reference;
    typedef T&       reference;
    
    ~ts_insert_bag() {
      while (head) {
	bag_item* n = const_cast<bag_item*>(head);
	head = n->next;
	delete n;
      }
    }

    class iterator {
      bag_item* b;
      friend class ts_insert_bag;
      iterator(bag_item* B) :b(B) {}
    public:
      typedef ptrdiff_t                 difference_type;
      typedef std::forward_iterator_tag iterator_category;
      typedef T                         value_type;
      typedef T*                        pointer;
      typedef T&                        reference;

      iterator(const iterator& rhs) {
	b = rhs.b;
      }
      iterator() :b(0) {}

      void incr() { b = b->next; }
      bool operator==(const iterator& rhs) const { return b == rhs.b; }
      bool operator!=(const iterator& rhs) const { return b != rhs.b; }

      reference operator*()  const { return b->item; }
      pointer   operator->() const { return &(operator*()); }
      iterator& operator++() { b = b->next; return *this; }
      iterator  operator++(int) { iterator __tmp = *this; b = b->next; return __tmp; }
    };

    iterator begin() {
      return iterator(const_cast<bag_item*>(head));
    }

    iterator end() {
      return iterator(0);
    }

    //Only this is thread safe
    reference push(const T& val) {
      bag_item* B = new bag_item(val);
      i_push(B);
      return B->item;
    }

  };
}
