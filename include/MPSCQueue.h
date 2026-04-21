#pragma once 
#include <type_traits>
#include <atomic>
#include <concepts>
#include <array>
#include <utility>


template <typename T, typename CAPACITY>
class MPSCQueue {
    static_assert((CAPACITY > 0) && ((CAPACITY & (CAPACITY - 1) == 0),
                  "CAPACITY must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "Must be is_trivially_copyable_v");
    
    static constexpr std::size_t MASK = CAPACITY - 1;
    // head tail and all seqs are monotonically increasing
    
    static constexpr int PAD = 64
        - static_cast<int>(sizeof(std::atomic<uint32_t))
        - static_cast<int>(sizeof(T));

    //Each slot has a sequence number that encodes its state:
    //   seq == index:         slot is empty, ready to be claimed by a producer
    //   seq == index + 1:     slot is being written (reserved, not yet committed)
    //   seq == index + CAPACITY: slot is full, ready to be read by consumer
    struct alignas(64) Slot {
        std::atomic<uint32_t> seq{0};
        T                     data;
        // Pad to 64 bytes to prevent false sharing between adjacent slots.
        // Without padding, two producers writing adjacent slots share a cache
        // line ΓÇö their seq writes invalidate each other's L1 lines.
        uint8_t _pad[PAD > 0 ? PAD : 1];
    };

    alignas(64) std::atomic<std::size_t> head{0};
    alignas(64) std::atomic<std::size_t> tail{0};
    alignas(64) std::array<Slot, CAPACITY> slots;
    

public:

    MPSCQueue() {
        for (uint32_t i{0}; i < CAPACITY; ++i) {
            this->slots[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    // if seq == localTail -> slot is free 
    // if seq < localTail  -> queue is full 
    // if seq > localTail -> another producer claimed, retry!
    // Retuns false if the queue is full so caller should do something 
    // ie (stop re-arming EPOLLIN / stop re-arming recv SQEs)
    bool push(const T& item) noexcept {

        while (true) {
            std::size_t localTail = this->tail.load(std::memory_order_relaxed);
            Slot& slot = this->slots[localTail & MASK];
            uint32_t curr_seq = slot.seq.load(std::memory_order_acquire); 
            int32_t diff = static_cast<int32_t>(curr_seq) - static_cast<int32_t>(localTail);
            
            if (diff == 0) {
                uint32_t claimed_slot = localTail;
                if (this->tail.compare_exchange_weak(localTail, localTail + 1,
                                                     std::memory_order_relaxed)) {

                    slot.data = item;
                    // release to ensure visibility before buffer write 
                    slot.seq.store(claimed_slot + 1, std::memory_order_release);

                    return true;
                }

                // CAS failed retry 
            }

            else if (diff < 0) {
                //queue is full 
                return false;
            }

            // diff > 0 slot taken by another producer so retry
        }

        return false;
        
    } 
    
    // called by single consumer GPU dispatch thread 
    // returns false if queue is empty
    bool pop(T& out) noexcept {
        std::size_t localHead = this->head.load(std::memory_order_relaxed);
        Slot& slot = this->slots[localHead & MASK];
        uint32_t curr_seq = slot.seq.load(std::memory_order_acquire);
        int32_t diff = static_cast<int32_t>(curr_seq) - static_cast<int32_t>(localHead + 1);

        if (diff == 0) {
            out = slot.data;
            // release so producers can read new head 
            this->head.store(localHead + 1, std::memory_order_relaxed);
            slot.seq.store(localHead + CAPACITY, std::memory_order_release);
            return true;
        }

        // empty
        return false;
    }

    uint32_t size() const noexcept {
        uint32_t localHead = this->head.load(std::memory_order_relaxed);
        uint32_t localTail = this->tail.load(std::memory_order_relaxed);
        return localTail - localHead;
    }

    bool empty() const noexcept { return size() == 0; }

    static constexpr uint32_t capacity() { return CAPACITY; }
};
