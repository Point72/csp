#ifndef _IN_CSP_CORE_SRMWLOCKFREEQUEUE_H
#define _IN_CSP_CORE_SRMWLOCKFREEQUEUE_H

#include <csp/core/System.h>
#include <csp/core/Time.h>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace csp
{

/*
  The SRMWLockFreeQueue is safe for multipe writers / single reader access.  Likely not the most efficient impl, but certainly
  one of the simplest!  template type is required to have an intrinsic next pointer
*/
template< typename T >
class alignas(CACHELINE_SIZE) SRMWLockFreeQueue
{
public:
    class Batch
    {
    public:
        Batch() : m_head( nullptr ), m_tail( nullptr ) {}

        void append( T * event )
        {
            //note we link the batch events in reverse order so that we can do a quick CAS switcheroo
            if( !m_head )
                m_head = m_tail = event;
            else
            {
                event -> next = m_head;
                m_head = event;
            }
        }

        void clear()
        {
            m_head = m_tail = nullptr;
        }

        bool empty() const { return m_head == nullptr; }

    protected:
        T * last() { return m_head; }

    private:
        T * m_head;
        T * m_tail;

        friend class SRMWLockFreeQueue;
    };

    SRMWLockFreeQueue( bool blocking = false ) : m_blocking( blocking ), m_head( nullptr )
    {}
    
    ~SRMWLockFreeQueue() {}

    bool empty() const { return m_head == nullptr && m_curItems == nullptr; }
    void push( T * );
    //atomic push, batch will be cleared after this call
    void push( Batch & batch ); 

    //pop calls can return NULL if empty
    //pop a single item
    T * pop( TimeDelta maxWait = TimeDelta() );

    T * peek();

    //pop all pending items, need to iterate over T -> next
    T * popAll( TimeDelta maxWait = TimeDelta() );

    void wait( TimeDelta maxWait )
    {
        if( !m_blocking || maxWait.asNanoseconds() <= 0 )
            return;

        std::unique_lock<std::mutex> lock( m_lock );
        if( !m_head.load( std::memory_order_relaxed ) )
            m_condition.wait_for( lock, std::chrono::nanoseconds( maxWait.asNanoseconds() ), [this]() { return m_head.load() != nullptr ; } );
    }

private:

    //These arent used if we are busy looping
    bool                        m_blocking;
    std::mutex                  m_lock;
    std::condition_variable     m_condition;
    
    std::atomic<T *>            m_head;
    alignas(CACHELINE_SIZE) T * m_curItems;
};

template< typename T >
inline void SRMWLockFreeQueue<T>::push( T * item )
{
    //ABA problem?  Not sure, even if head pointer did go through an ABA cycle
    //it seems like the queue will still stay stable
    T * old_head = m_head.load( std::memory_order_relaxed );
    item -> next = old_head;
    while( !m_head.compare_exchange_weak( item -> next, item, std::memory_order_release ) ) 
    {}

    if( m_blocking && old_head == nullptr )
        m_condition.notify_one();
}

template< typename T >
inline void SRMWLockFreeQueue<T>::push( Batch & batch )
{
    //Batch already gaurantees events are in the correct reverse ordering
    //so we just set m_head to batch head and link batch tail to previous head
    T * old_head = m_head.load( std::memory_order_relaxed );
    batch.m_tail -> next = old_head;
    while( !m_head.compare_exchange_weak( batch.m_tail -> next, batch.m_head, std::memory_order_release ) ) 
    {}

    batch.clear();

    if( m_blocking && old_head == nullptr )
        m_condition.notify_one();
}

template< typename T >
inline T * SRMWLockFreeQueue<T>::pop( TimeDelta maxWait )
{
    //current impl has push creating a linked list in reverse order
    //we want to maintian a single popEvent interface for future impls, so 
    //we popAll into m_curItems and work off of that as long its available

    //single reader thread here
    if( !m_curItems )
        m_curItems = popAll( maxWait );

    if( m_curItems )
    {
        T * ret = m_curItems;
        m_curItems = m_curItems -> next;
        return ret;
    }

    return nullptr;
}

template< typename T >
inline T * SRMWLockFreeQueue<T>::peek()
{
    if( !m_curItems )
        m_curItems = popAll();
    return m_curItems;
}

template< typename T >
inline T * SRMWLockFreeQueue<T>::popAll( TimeDelta maxWait )
{
    if( m_blocking )
        wait( maxWait );

    T * head = m_head.exchange( nullptr );
    T * prev = nullptr;

    while( head )
    {
        T * next = head -> next;
        head -> next = prev;
        prev = head;
        head = next;
    }

    return prev;    
}

}

#endif
