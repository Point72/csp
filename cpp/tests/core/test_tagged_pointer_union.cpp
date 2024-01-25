#include <gtest/gtest.h>
#include <csp/core/TaggedPointerUnion.h>
#include <memory>

using namespace csp;

TEST( TaggedPointerUnion, test_basic_functionality )
{
    using TP2 = TaggedPointerUnion<int,double>;
    TP2 tp2;
    
    ASSERT_EQ( TP2::NUM_TAGS, 2u );
    ASSERT_EQ( TP2::TAG_BITS, 2u );
    ASSERT_EQ( TP2::TAG_MASK, 3u );

    ASSERT_EQ( TP2::typeBit<int>(),    0u );
    ASSERT_EQ( TP2::typeBit<double>(), 1u );

    ASSERT_EQ( tp2.raw(), nullptr );

    std::unique_ptr<int> i( new int{ 5 } );
    std::unique_ptr<double> d( new double{ 123.456 } );

    tp2.set( d.get() );

    ASSERT_TRUE( tp2.isSet<double>() );
    ASSERT_FALSE( tp2.isSet<int>() );
    ASSERT_EQ( tp2.get<double>(), d.get() );

    tp2.set( i.get() );
    ASSERT_TRUE( tp2.isSet<int>() );
    ASSERT_FALSE( tp2.isSet<double>() );
    ASSERT_EQ( tp2.get<int>(), i.get() );
}
