#include <gtest/gtest.h>

#include <csp/core/QueueWaiter.h>

#include <atomic>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/select.h>
#include <unistd.h>
#endif

using namespace csp;

namespace
{

bool isReadable( FdWaiter & waiter )
{
    fd_set readSet;
    FD_ZERO( &readSet );
    FD_SET( waiter.readFd(), &readSet );

    struct timeval timeout;
    timeout.tv_sec  = 0;
    timeout.tv_usec = 0;

    return select( ( int ) waiter.readFd() + 1, &readSet, nullptr, nullptr, &timeout ) > 0;
}

// clear() drains a bounded amount per call, so a saturated buffer needs several passes
bool drainUntilQuiet( FdWaiter & waiter, int maxPasses = 10000 )
{
    for( int i = 0; i < maxPasses; ++i )
    {
        if( !isReadable( waiter ) )
            return true;
        waiter.clear();
    }
    return !isReadable( waiter );
}

}

TEST( FdWaiterTest, quiet_until_notified )
{
    FdWaiter waiter;
    ASSERT_TRUE( waiter.isValid() );
    EXPECT_FALSE( isReadable( waiter ) );
}

TEST( FdWaiterTest, notify_makes_readable )
{
    FdWaiter waiter;
    waiter.notify();
    EXPECT_TRUE( isReadable( waiter ) );
}

TEST( FdWaiterTest, clear_makes_quiet )
{
    FdWaiter waiter;
    waiter.notify();
    waiter.clear();
    EXPECT_FALSE( isReadable( waiter ) );
}

// A notify after a clear must signal again.  A sticky "already notified" flag would swallow it and
// strand the consumer in select().
TEST( FdWaiterTest, notify_after_clear_signals_again )
{
    FdWaiter waiter;

    for( int i = 0; i < 5; ++i )
    {
        waiter.notify();
        EXPECT_TRUE( isReadable( waiter ) ) << "iteration " << i;
        waiter.clear();
        EXPECT_FALSE( isReadable( waiter ) ) << "iteration " << i;
    }
}

// notify() is unsynchronized and writes on every call, so the underlying pipe/counter can saturate.
// Saturation must neither block nor wedge the waiter, and the waiter must drain back to quiet.
TEST( FdWaiterTest, saturation_is_handled )
{
    FdWaiter waiter;

    for( int i = 0; i < 200000; ++i )
        waiter.notify();

    EXPECT_TRUE( isReadable( waiter ) );

    EXPECT_TRUE( drainUntilQuiet( waiter ) );

    waiter.notify();
    EXPECT_TRUE( isReadable( waiter ) );
}

// Regression test.  The previous implementation suppressed notify() whenever an internal
// "already notified" flag was set, and only clear() reset it.  Anything that drained the fd by
// other means - a selector reading it directly, a partial read, a failed write leaving the flag
// set - stranded the flag and silently swallowed every subsequent notify.
TEST( FdWaiterTest, notify_recovers_if_fd_drained_externally )
{
    FdWaiter waiter;

    waiter.notify();
    ASSERT_TRUE( isReadable( waiter ) );

    // Drain behind the waiter's back, without going through clear()
    char buf[ 64 ];
#ifdef _WIN32
    while( recv( waiter.readFd(), buf, sizeof( buf ), 0 ) > 0 ) {}
#else
    while( ::read( waiter.readFd(), buf, sizeof( buf ) ) > 0 ) {}
#endif
    ASSERT_FALSE( isReadable( waiter ) );

    waiter.notify();
    EXPECT_TRUE( isReadable( waiter ) );
}

TEST( FdWaiterTest, concurrent_notify )
{
    FdWaiter waiter;

    std::vector<std::thread> threads;
    for( int t = 0; t < 8; ++t )
        threads.emplace_back( [&waiter]() { for( int i = 0; i < 1000; ++i ) waiter.notify(); } );

    for( auto & thread : threads )
        thread.join();

    EXPECT_TRUE( isReadable( waiter ) );

    EXPECT_TRUE( drainUntilQuiet( waiter ) );
}

// notify() is unsynchronized, so producers can refill the fd while clear() is draining it.
// clear() must be bounded and return regardless; if it spun until the fd ran dry this would hang.
TEST( FdWaiterTest, clear_terminates_while_producers_are_running )
{
    FdWaiter waiter;

    std::atomic<bool> stop{ false };
    std::vector<std::thread> threads;
    for( int t = 0; t < 4; ++t )
        threads.emplace_back( [&waiter, &stop]() { while( !stop.load( std::memory_order_relaxed ) ) waiter.notify(); } );

    for( int i = 0; i < 1000; ++i )
        waiter.clear();

    stop.store( true, std::memory_order_relaxed );
    for( auto & thread : threads )
        thread.join();

    SUCCEED();
}
