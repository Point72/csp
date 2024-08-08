#include <gtest/gtest.h>
#include <csp/core/Platform.h>

TEST( Platform, clz )
{
    uint8_t x = 1;
    for( int i = 7; i >= 0; --i, x <<= 1 )
        ASSERT_EQ( clz( x ), ( uint8_t ) i );

    uint16_t y = 1;
    for( int i = 15; i >= 0; --i, y <<= 1 )
        ASSERT_EQ( clz( y ), ( uint16_t ) i );

    uint32_t z = 1;
    for( int i = 31; i >= 0; --i, z <<= 1 )
        ASSERT_EQ( clz( z ), ( uint32_t ) i );

    uint64_t w = 1;
    for( int i = 63; i >= 0; --i, w <<= 1 )
        ASSERT_EQ( clz( w ), ( uint64_t ) i );
}

TEST( Platform, ffs )
{
    uint8_t x = 1;
    for( int i = 1; i <= 8; ++i, x <<= 1 )
        ASSERT_EQ( ffs( x ), i );
    ASSERT_EQ( ffs( x ), 0 );

    uint16_t y = 1;
    for( int i = 1; i <= 16; ++i, y <<= 1 )
        ASSERT_EQ( ffs( y ), i );
    ASSERT_EQ( ffs( y ), 0 );

    uint32_t z = 1;
    for( int i = 1; i <= 32; ++i, z <<= 1 )
        ASSERT_EQ( ffs( z ), i );
    ASSERT_EQ( ffs( z ), 0 );

    uint64_t w = 1;
    for( int i = 1; i <= 64; ++i, w <<= 1 )
        ASSERT_EQ( ffs( w ), i );
    ASSERT_EQ( ffs( w ), 0 );
}