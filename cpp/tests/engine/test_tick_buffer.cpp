#include <csp/core/Exception.h>
#include <csp/engine/TickBuffer.h>
#include <gtest/gtest.h>

TEST( TickBufferTest, test_basic_functionality )
{
    csp::TickBuffer<double> buffer;
    buffer.growBuffer( 3 );
    ASSERT_EQ( buffer.capacity(), 3 );

    for( int i = 0; i < 13; ++i )
    {
        buffer.push_back( i );
        ASSERT_EQ( buffer.numTicks(), std::min( i + 1, 3 ) ) << i;
        ASSERT_EQ( buffer.valueAtIndex( 0 ), i );

        for( int j = 0; j < std::min( i, 3 ); ++j )
            ASSERT_EQ( buffer.valueAtIndex( j ), i - j );
    }
}

TEST( TickBufferTest, test_resize )
{
    csp::TickBuffer<double> buffer;
    buffer.growBuffer( 3 );
    ASSERT_EQ( buffer.capacity(), 3 );

    buffer.push_back( 1 );
    buffer.push_back( 2 );

    //resize before full
    buffer.growBuffer( 5 );
    buffer.push_back( 3 );

    ASSERT_EQ( buffer.numTicks(), 3 );
    ASSERT_EQ( buffer.capacity(), 5 );
    ASSERT_EQ( buffer.valueAtIndex( 0 ), 3 );
    ASSERT_EQ( buffer.valueAtIndex( 1 ), 2 );
    ASSERT_EQ( buffer.valueAtIndex( 2 ), 1 );
    ASSERT_THROW( buffer.valueAtIndex( 3 ), csp::RangeError ); //exception

    buffer.push_back( 4 );
    buffer.push_back( 5 );
    buffer.push_back( 6 );
    for( int i = 0; i < 5; ++i )
        ASSERT_EQ( buffer[ i ], 6 - i ) << i;

    //resize after full
    buffer.growBuffer( 8 );
    ASSERT_EQ( buffer.numTicks(), 5 );
    ASSERT_EQ( buffer.capacity(), 8 );

    for( int i = 0; i < 5; ++i )
        ASSERT_EQ( buffer[ i ], 6 - i );
    ASSERT_THROW( buffer.valueAtIndex( 5 ), csp::RangeError ); //exception

    buffer.push_back( 7 );
    ASSERT_EQ( buffer.valueAtIndex( 0 ), 7 );
    //TODO: THIS TEST IS BROKEN, fix it
    ASSERT_EQ( buffer.valueAtIndex( 5 ), 2 );
}

TEST( TickBufferTest, test_flatten )
{
    int * values_wrap, * values_nowrap, * values_single;
    {
        csp::TickBuffer<int> buffer;
        buffer.growBuffer( 5 );
        ASSERT_EQ( buffer.capacity(), 5 );

        buffer.push_back( 0 );

        values_single = buffer.flatten( 0, 0 );

        buffer.push_back( 1 );
        buffer.push_back( 2 );
        buffer.push_back( 3 );
        buffer.push_back( 4 );

        values_nowrap = buffer.flatten( 4, 0 );

        buffer.push_back( 5 );
        buffer.push_back( 6 );
        buffer.push_back( 7 );

        for( int i = 0; i < 5; ++i )
            ASSERT_EQ( buffer[ i ], 7 - i );

        values_wrap = buffer.flatten( 4, 0 );

        ASSERT_THROW( buffer.flatten( 5, 0 ), csp::RangeError );
    }

    ASSERT_EQ( values_single[ 0 ], 0 );

    for( int i = 0; i < 5; ++i )
    {
        ASSERT_EQ( values_wrap[ i ], i + 3 );
        ASSERT_EQ( values_nowrap[ i ], i );
    }

    free( values_wrap );
    free( values_nowrap );
    free( values_single );
}
