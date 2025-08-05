#include <gtest/gtest.h>
#include <csp/core/DynamicBitSet.h>

#include <algorithm>
#include <unordered_set>
#include <random>
#include <vector>

using namespace csp;

void testIteration( const std::vector<int> & vec, const DynamicBitSet<> & bitset  )
{
    size_t i = 0;
    for( auto it = bitset.find_first(); it != DynamicBitSet<>::npos; i++, it = bitset.find_next( it ) )
        ASSERT_EQ( it, vec[i] );

    ASSERT_EQ( i, vec.size() );

    for( auto it = bitset.find_last(); it != DynamicBitSet<>::npos; i--, it = bitset.find_prev( it ) )
        ASSERT_EQ( it, vec[i-1] );

    ASSERT_EQ( i, 0ul );
}

TEST( DynamicBitSetTest, basic_functionality )
{
    std::vector<std::pair<size_t, size_t>> test_cases = { std::make_pair( 4096, 400 ), std::make_pair( 200, 50 ), std::make_pair( 100, 100 ), std::make_pair( 10, 0 ) };

    for( auto & test_case : test_cases )
    {
        auto size = test_case.first;
        auto num_set = test_case.second;

        auto bitset = DynamicBitSet<>( size );
        std::unordered_set< int > vals;
        while( vals.size() < num_set )
            vals.insert( rand() % size );
        
        for( auto i : vals )
            bitset.set( i );

        std::vector< int > vec;
        for( auto i : vals ) vec.push_back( i );
        std::sort( vec.begin(), vec.end() );

        testIteration( vec, bitset );

        std::vector< int > half_vec;
        for( size_t j = 0; j < vec.size(); j+=2 )
        {
            bitset.reset( vec[j] );
            half_vec.push_back( vec[j+1] );
        }
        std::sort( half_vec.begin(), half_vec.end() );

        testIteration( half_vec, bitset );

        for( size_t j = 1; j < vec.size(); j+=2 )
            bitset.reset( vec[j] );

        ASSERT_EQ( bitset.find_first(), DynamicBitSet<>::npos );
        ASSERT_EQ( bitset.find_last(), DynamicBitSet<>::npos );
    }
}

TEST( DynamicBitSetTest, constructors )
{
    DynamicBitSet<> bitset; // default construct
    ASSERT_EQ( bitset.size(), 0ul );
    bitset = DynamicBitSet<>( 10 ); // move assignment
    ASSERT_EQ( bitset.size(), 10ul );
    bitset.set( 5 );
    bitset.set( 7 );
    DynamicBitSet<> bitset2( std::move( bitset ) ); // move constructor
    ASSERT_EQ( bitset2.find_first(), 5 );
    ASSERT_EQ( bitset2.find_last(), 7 );
}