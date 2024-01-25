#include <gtest/gtest.h>
#include <csp/core/SRMWLockFreeQueue.h>
#include <future>

using namespace csp;

struct TestEvent
{
    TestEvent( int v_ ) : v( v_ ), next( nullptr ) {}
    int v;

    TestEvent * next;
};


TEST( SRMWLockFreeQueueNoThreads, basic_functionality )
{
    SRMWLockFreeQueue<TestEvent> q;
    int x = 0;
    for( int n = 0; n < 10; ++n )
    {
        for( int i = 0; i < 100; ++i )
            q.push( new TestEvent( n * 100 + i ) );

        TestEvent * evt;
        while( ( evt = ( TestEvent * ) q.pop() ) )
        {
            ASSERT_EQ( evt -> v, x++ );
            delete evt;
        }
    }
}

int64_t pushEvents( SRMWLockFreeQueue<TestEvent> * q, int tickCount, int id )
{
    int64_t sum = 0;
    for( int i = 0; i < tickCount; ++i )
    {
        q -> push( new TestEvent( tickCount * id + i ) );
        sum += tickCount * id + i;
    }

    return sum;
}

int64_t pushBatchEvents( SRMWLockFreeQueue<TestEvent> * q, int tickCount, int batchsize, int id )
{
    int64_t sum = 0;
    for( int i = 0; i < tickCount; i += batchsize )
    {
        {
            SRMWLockFreeQueue<TestEvent>::Batch batch;
            for( int j = 0; j < batchsize; ++j )
            {
                batch.append( new TestEvent( tickCount * id + i + j ) );
                sum += tickCount * id + i + j;
            }

            q -> push( batch );
        }

    }

    return sum;
}

TEST( SRMWLockFreeQueueWThreads, basic_functionality )
{
    //test multiple writers competing
    int num_threads = 16;
    int tickCount = 1000000;
    SRMWLockFreeQueue<TestEvent> q;

    std::vector<std::future<int64_t> > res;
    for( int i = 0; i < num_threads; ++i )
        res.push_back( std::async( std::launch::async, pushEvents, &q, tickCount, i ) );

    int64_t c = 0;
    int64_t sum = 0;
    while( c < num_threads * tickCount )
    {
        TestEvent * e = ( TestEvent * ) q.pop();
        if( e )
        {
            ++c;
            sum += e -> v;
            delete e;
        }
    }

    int64_t expected = 0;
    for( auto & r : res )
        expected += r.get();

    ASSERT_EQ( sum, expected );

}

TEST( SRMWLockFreeQueueWThreads, batch_functionality )
{
    //test multiple writers competing
    int num_threads = 16;
    int tickCount = 1000000;
    SRMWLockFreeQueue<TestEvent> q;

    std::vector<std::future<int64_t> > res;
    for( int i = 0; i < num_threads; ++i )
        res.push_back( std::async( std::launch::async, pushBatchEvents, &q, tickCount, 20, i ) );

    int64_t c = 0;
    int64_t sum = 0;
    while( c < num_threads * tickCount )
    {
        TestEvent * e = ( TestEvent * ) q.pop();
        if( e )
        {
            ++c;
            sum += e -> v;
            delete e;
        }
    }

    int64_t expected = 0;
    for( auto & r : res )
        expected += r.get();

    ASSERT_EQ( sum, expected );

}

TEST( SRMWLockFreeQueueNoThreads, test_empty )
{
    SRMWLockFreeQueue<TestEvent> q;
    for( int i = 0; i < 100; ++i )
        q.push( new TestEvent(  i ) );
    ASSERT_FALSE(q.empty());
    delete q.pop();
    ASSERT_FALSE(q.empty());

    while( !q.empty() )
        delete q.pop();
}

void testBlockingWait( TimeDelta maxWait )
{
    SRMWLockFreeQueue<TestEvent> q( true );
    int nEvents = 1000;

    auto result = std::async( std::launch::async, [&](){
        int eventsRecvd = 0;
        int total = 0;
        while( eventsRecvd < nEvents )
        {
            for( auto * e = q.pop( maxWait ); e != nullptr; e = q.pop( maxWait ) )
            {
                ++eventsRecvd;
                total += e -> v;
                delete e;
            }
        }
        return total;
    });

    int expectedResult = 0;
    for( int i = 0; i < nEvents; ++i )
    {
        expectedResult += i;
        q.push( new TestEvent( i ) );
    }

    ASSERT_EQ( expectedResult, result.get() );
}

TEST( SRMWLockFreeQueueWThreads, blocking_wait )
{
    testBlockingWait( TimeDelta::NONE() );
    testBlockingWait( TimeDelta::fromMilliseconds( 200 ) );
    testBlockingWait( TimeDelta::fromNanoseconds( 1 ) );
}
