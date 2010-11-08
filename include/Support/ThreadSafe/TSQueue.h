// Queue implementation -*- C++ -*-

#include <deque>
#include <algorithm>
#include <ext/malloc_allocator.h>

namespace threadsafe {
  
  template< class _Tp, class _Lock = simpleLock >
  class ts_queue {
    typedef std::deque<_Tp, __gnu_cxx::malloc_allocator<_Tp> > _Sequence;
  public:
    typedef typename _Sequence::value_type                value_type;
    typedef typename _Sequence::const_reference           const_reference;
    typedef typename _Sequence::size_type                 size_type;
    typedef          _Sequence                            container_type;
    
  protected:
    //  See queue::c for notes on this name.
    _Sequence c;
    mutable _Lock lock;


  public:

    explicit
    ts_queue(const _Sequence& __c)
      : c(__c) { }
    
    ts_queue()
      : c() { }
    
    /**
     *  Returns true if the %stack is empty.
     */
    bool
    empty() const
    { 
      lock.read_lock();
      bool retval = c.empty();
      lock.read_unlock();
      return retval;
    }

    template<typename S>
    bool moveTo(S& other, int count = 0) {
      bool retval = false;
      lock.write_lock();
      bool do_all = count == 0;
      while (!c.empty() && (do_all || count)) {
	value_type v = c.front();
	c.pop_front();
	other.push(v);
	retval = true;
	--count;
      }
      lock.write_unlock();
      return retval;
    }

    template<typename Siter>
    void insert(Siter b, Siter e) {
      lock.write_lock();
      for (; b != e; ++b)
	c.push(*b);
      lock.write_unlock();
    }

    /**  Returns the number of elements in the %stack.  */
    size_type
    size() const
    { 
      lock.read_lock();
      size_type retval = c.size();
      lock.read_unlock();
      return retval;
    }

    /**
     *  @brief  Add data to the top of the %stack.
     *  @param  x  Data to be added.
     *
     *  This is a typical %stack operation.  The function creates an
     *  element at the top of the %stack and assigns the given data
     *  to it.  The time complexity of the operation depends on the
     *  underlying sequence.
     */
    void
    push(const value_type& __x)
    {
      lock.write_lock();
      c.push_back(__x);
      lock.write_unlock();
    }

    /**
     *  @brief  Removes first element and returns it
     *
     *  This is a typical %stack operation.  It shrinks the %stack
     *  by one.  The time complexity of the operation depends on the
     *  underlying sequence.
     *
     */
    value_type
    pop(bool& suc)
    {
      lock.read_lock();
      value_type retval;
      if (!c.empty()) {
	lock.promote();
	retval = c.front();
	c.pop();
	lock.write_unlock();
	suc = true;
      } else {
	lock.read_unlock();
	suc = false;
      }
      return retval;
    }

    value_type
    peek(bool& suc) {
      lock.read_lock();
      value_type retval;
      if (!c.empty()) {
	lock.promote();
	retval = c.front();
	lock.write_unlock();
	suc = true;
      } else {
	lock.read_unlock();
	suc = false;
      }
      return retval;
    }      

  void sort() {
    lock.write_lock();
    std::sort(c.begin(), c.end());
    lock.write_unlock();
  }

  };
}
