/*
  thread_work.h

  Copyright (C) 20XX--20XX, The University of Texas at Austin

  Author: Vishwesh Jatala  <vishwesh.jatala@austin.utexas.edu>
*/

struct ThreadWork {

	PipeContextT<Worklist2> thread_work_wl;
	PipeContextT<Worklist2> thread_src_wl;
	Shared<int> thread_prefix_work_wl;
	bool initialized = false;

	void init_thread_work(int size) {
		if (!initialized) {
			thread_work_wl = PipeContextT<Worklist2>(size);
			thread_src_wl = PipeContextT<Worklist2>(size);

			thread_prefix_work_wl.alloc(size);
			thread_prefix_work_wl.zero_gpu();
			initialized = true;
		}
	}

	void compute_prefix_sum() {

		cub::CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory
		// Determine temporary device storage requirements for inclusive prefix sum
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;

		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, thread_work_wl.in_wl().dwl,
				thread_prefix_work_wl.gpu_wr_ptr(), thread_work_wl.in_wl().nitems());
		// Allocate temporary storage for inclusive prefix sum
		CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
		// Run inclusive prefix sum
		cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, thread_work_wl.in_wl().dwl,
				thread_prefix_work_wl.gpu_wr_ptr(), thread_work_wl.in_wl().nitems());
	}

	void reset_thread_work() {
		thread_prefix_work_wl.zero_gpu();
		thread_work_wl.in_wl().reset();
		thread_src_wl.in_wl().reset();
	}

};

__device__ unsigned compute_src_and_offset(unsigned first, unsigned last, unsigned index, int * thread_prefix_work_wl,
			unsigned num_items, unsigned int &offset) {

		unsigned middle = (first + last)/2;

		if( index <= thread_prefix_work_wl[first] ) {
			if(first == 0 ) {
				offset =  index -1 ;
				return first;
			}
			else {
				offset =  index - thread_prefix_work_wl[first-1] -1 ;
				return first;
			}
		}
		while (first + 1 != last)
		{
		   middle = (first + last)/2;
		   if(index > thread_prefix_work_wl[middle])
		   {
			   first = middle ;
		   }
		   else {
			   last = middle;
		   }
		}
		offset =  index - thread_prefix_work_wl[first] - 1 ;
		return last;
	}
