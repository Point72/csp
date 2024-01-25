#include <gtest/gtest.h>
#include <csp/engine/TimeSeries.h>

using namespace csp;

struct Foo
{
    Foo( int a_, int b_ ) : a( a_ ), b( b_ )
    {}
    
    int a;
    int b;
};

TEST( TimeSeriesTest, test_basic_functionality )
{
    DateTime  time( 2017, 1, 1 );
    TimeDelta delta = TimeDelta::fromSeconds( 1 );
    
    for( int tickCountSize = 1; tickCountSize < 10; tickCountSize += 2 )
    {
        TimeSeriesTyped<double> ts_d;
        TimeSeries & ts_d_ref = ts_d;
        ts_d_ref.setTickCountPolicy( tickCountSize );
        
        for( int i =0; i < tickCountSize; ++i )
        {
            ts_d_ref.addTickTyped<double>( time + delta * i, i );
            EXPECT_EQ( ts_d_ref.lastValueTyped<double>(), i );
            EXPECT_EQ( ts_d_ref.lastTime(), time + delta * i );
            EXPECT_EQ( ts_d_ref.count(), i + 1 );
            EXPECT_EQ( ts_d_ref.numTicks(), i + 1 );

            for( int j = 0; j<=i; ++j )
            {
                EXPECT_EQ( ts_d_ref.valueAtIndex<double>( j ), i - j );
                EXPECT_EQ( ts_d_ref.timeAtIndex( j ), time + delta*(i - j ));
            }
        }

        EXPECT_THROW( ts_d_ref.valueAtIndex<double>( tickCountSize ), std::exception );
    }

    TimeSeriesTyped<bool>                  ts_b;
    TimeSeriesTyped<std::shared_ptr<Foo> > ts_s;

    ASSERT_EQ( ts_b.numTicks(), 0 );
    ASSERT_EQ( ts_b.count(), 0 );
    ASSERT_EQ( ts_s.numTicks(), 0 );
    ASSERT_EQ( ts_s.count(), 0 );

    ts_b.setTickCountPolicy( 10 );
    ts_s.setTickTimeWindowPolicy( TimeDelta::fromSeconds( 10 ) );

    ASSERT_EQ( ts_b.numTicks(), 0 );
    ASSERT_EQ( ts_b.count(), 0 );
    ASSERT_EQ( ts_s.numTicks(), 0 );
    ASSERT_EQ( ts_s.count(), 0 );

    for( int i = 0; i < 25; i++ )
    {
        ts_b.addTick( time + delta * i,     true );
    }
    
    ASSERT_EQ( ts_b.numTicks(), 10 );
    ASSERT_EQ( ts_b.count(), 25 );

    for( int i = 0; i < 25; i++ )
    {
        ts_s.addTick( time + delta * i,     std::make_shared<Foo>( 1, 2 ) );
    }

    ASSERT_EQ( ts_s.numTicks(), 16 ); // buffer gets resized to nextpow2, so 16 ticks are available
    ASSERT_EQ( ts_s.count(), 25 );
}
