#pragma once 
#include <atomic>
#include <cstdint>

// SPSCQueue<T, CAPACITY>: lock free single producer single consumer ring buffer.
//
// User for:
//  - fast batch queue: Classifier (producer) -> GPU dispatch (consumer)
//  - slow batch queue: same 
//  - match qeueue: CPU Matcher (producer) -> CPU Matcher consumer path 
//  - Fill return: GPU Dispatch (producer) -> Reactor thread (consumer)
//
//  CAPACITY must be a power of 2

template<typename T, uint32_t CAPACITY>
class SPSCQueue {
  static_assert((CAPACITY & (CAPACITY - 1)) == 0,
    "CAPACITY must be a power of 2");
  static_assert(std::is_trivially_copyable_v<T>);

  static constexpr uint32_t MASK = CAPACITY - 1;

  alignas(64) std::atomic<uint32_t> head{0};
  alignas(64) std::atomic<uint32_t> tail{0};
  alignas(64) std::array<T, CAPACITY> slots;

  public:
    // push: called by producer 
    // Returns false if full
    bool push(const T& item) noexcept {
      uint32_t tail = this->tail.load(std::memory_order_relaxed);
      // covers both in memory barrier i think
      uint32_t head = this->head.load(std::memory_order_acquire);
      
      // ONLY WORKS IF CAPACITY IS POWER OF 2 
      // (as when uint32_t overflows it goes to 0, ie 128 & 127 = 0
      // but 100 & 99 != 0, so the different between monotonic counters would change)
      if (tail - head >= CAPACITY) {
        return false; // full 
      }

      // if bigger it loops around--equals % CAPACITY 
      this->slots[tail & MASK] = item;
      // fair game since no data race 
      this->tail.store(tail + 1, std::memory_order_release);
      return true;
    }

    // pop: called by consumer 
    // Returns false if empty 
    bool pop(T& out) noexcept {
      uint32_t tail = this->tail.load(std::memory_order_relaxed);
      // covers both in memory barrier i think
      uint32_t head = this->head.load(std::memory_order_acquire);

      if (head == tail) {
        return false;
      }

      // write to input 
      out = this->slots[head & MASK];
      // fair game since no data race 
      this->head.store(head + 1, std::memory_order_release);
      return true;
    }

    // returns size as monotonic difference 
    uint32_t size() const noexcept {
      return this->tail.load(std::memory_order_relaxed)
           - this->head.load(std::memory_order_relaxed);
    }

    bool empty() const noexcept { return size() == 0; }
    bool full() const noexcept { return size() >= CAPACITY; }
    static constexpr uint32_t capacity() { return CAPACITY; }
};
