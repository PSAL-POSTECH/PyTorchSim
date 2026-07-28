#ifndef PairDelayQueue_H
#define PairDelayQueue_H
#include <cassert>
#include <cstdint>
#include <queue>
#include <string>

template <typename T>
class DelayQueue {
 public:
  DelayQueue() {}
  DelayQueue(std::string name, bool only_latency, int max_size)
      : m_only_latency(only_latency),
        m_name(name),
        m_interval(0),
        m_cycle(0),
        m_max_size(max_size),
        m_issued(false),
        m_size(0) {}
  DelayQueue(std::string name) : DelayQueue(name, false, -1) {}
  void push(T data, int delay);
  void push(T data, int delay, int interval);
  void pop();
  T top();
  int size() { return m_size; }
  bool arrived();
  bool queue_empty();
  bool full();
  void cycle();

 private:
  struct QueueEntry {
    T data;
    uint64_t finish_cycle = 0;
  };
  std::string m_name;
  int m_interval;
  uint64_t m_cycle;
  int m_size;
  int m_max_size;
  bool m_issued;
  bool m_only_latency;
  std::queue<QueueEntry> m_queue;
};
#endif