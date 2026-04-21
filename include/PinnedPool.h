#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cassert>
#include "OrderMessage.h"
#include "SPSCQueue.h"


// CudaHostDeleter: a stateless deleter for the pinned memory in pool 
struct CudaHostDeleter {
  constexpr void operator()(OrderMessage* ptr) const {
    if (ptr) cudaFreeHost(ptr);
  }
};

// PinnedPool: a pool of pinned (cudaMallocHost) buffer slots.
//
// Each reactor owns one PinnedPool. When the reactor's epoll fires 
// recv-ready on a socket, it claims a slot from the pool, reads the
// OrderMessage into that slot, and pushes an OrderRef (containing reactor id)
// into the MPSC Queue 
//
// After GPU processes the batch containing the slot, CUDA stream callback
// fires and returns slot to pool 
//
// Timeline
//
//  slot is FREE -> reactor claims 
//  slot is IN_USE -> reactor writes OrderMessage into it + push OrderRef into MPSC 
//
//  slot is IN_QUEUE -> GPU dispatch thread holds the OrderRef 
//  slot is ON_GPU -> GPU kernel is reading pinned memory 
//  slot is FREE -> CUDA callback fires 
//
//  The free list is an SPSC queues
//    Producer: CUDA callback thread (returns freed slots) 
//    Consumer: reactor thread (claims slots for new receives)

static constexpr uint32_t POOL_SLOTS_PER_REACTOR = 512;

class PinnedPool {
  public:
    PinnedPool() = default;
    
    // called once before use
    void init(uint32_t reactor_id) {
      this->reactor_id = reactor_id;
      OrderMessage* raw_ptr = nullptr;
      cudaError_t err = cudaMallocHost(
          reinterpret_cast<void**>(&raw_ptr),
          POOL_SLOTS_PER_REACTOR * sizeof(OrderMessage)
      );
      assert(err == cudaSuccess && "cudaMallocHost failed for PinnedPool");
      
      this->base.reset(raw_ptr);
      
      // fill free list at start 
      for (uint32_t i{0}; i < POOL_SLOTS_PER_REACTOR; ++i) {
        bool ok = this->free_list.push(i);
        assert(ok);
      }
    }
    
    // called by reactor before a recv()
    // Returns -1 if the bool is exhausted (STOP ARMING EPOLLIN)
    // Else returns slot
    int acquire_slot() noexcept {
      uint32_t slot_id{};
      if (this->free_list.pop(slot_id)) {
        return static_cast<int>(slot_id);
      }
      return -1;
    }
  
    // slot_ptr: returns a pointer to the OrderMessage at the given slot.
    // This pointer is valid for both CPU writes and GPU reads
    // raw pointer should be fine since unique_ptr owns the pool
    OrderMessage* slot_ptr(uint32_t slot_id) noexcept {
      assert(slot_id < POOL_SLOTS_PER_REACTOR);
      return this->base.get() + slot_id;
    }

    // return_slot: called by the CUDA stream callback thread after GPU
    // is done reading the buffer slot.
    // thread safe since GPU callback thread is sole caller in SPSC 
    void return_slot(uint32_t slot_id) noexcept {
      assert(slot_id < POOL_SLOTS_PER_REACTOR);
      bool ok = this->free_list.push(slot_id);
      assert(ok && "PinnedPool free_list overflow -- something wrong happened");
    }

    // which reactor the pool is tied to 
    uint32_t reactor_id() const noexcept { return this->reactor_id; }

    // approx free list size 
    uint32_t available() const noexcept { return this->free_list.size(); }

  private:
    std::unique_ptr<OrderMessage[], CudaHostDeleter> base = nullptr;
    // SPSC free list 
    //  Producer: CUDA callback thread (returns slots after GPU done)
    //  Consumer: reactor thread (claims slot for recv)
    SPSCQueue<uint32_t, POOL_SLOTS_PER_REACTOR> free_list;
    uint32_t reactor_id;

};
