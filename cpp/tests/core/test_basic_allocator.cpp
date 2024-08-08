#include <gtest/gtest.h>
#include <csp/core/BasicAllocator.h>
#include <stdlib.h>

#ifdef WIN32
#define random rand
#endif

using namespace csp;

struct Foo
{
    int a;
    double b;
    char c;
};

TEST( BasicAllocator, basic_functionality )
{
    std::vector<size_t> allocCounts = { 32, 128, 999 };
    const size_t blockSize = 32;

    for( auto allocCount : allocCounts )
    {
        printf( "Running allocCount %lu...\n", allocCount );
        bool grow = allocCount > blockSize;
        TypedBasicAllocator<Foo> allocator( blockSize, false, grow );
        
        std::vector<Foo *> foos;
        foos.resize( allocCount );
        
        for( size_t i = 0; i < allocCount; ++i )
        {
            foos[ i ] = allocator.allocate();
            ASSERT_NE( foos[i], nullptr );
            foos[i] -> a = i;
            foos[i] -> b = i;
            foos[i] -> c = ( char ) i;
        }
        
        if( !grow )
        {
            ASSERT_EQ( allocator.allocate(), nullptr );
        }
        
        for( int i = 0; i < 100000; ++i )
        {
            int idx = random() % allocCount;
            if( foos[idx] )
            {
                //printf( "RBA to free idx: %d\n", idx );
                
                ASSERT_EQ( foos[idx] -> a, idx );
                ASSERT_EQ( foos[idx] -> b, idx );
                ASSERT_EQ( foos[idx] -> c, ( char ) idx );
                
                allocator.free( foos[idx] );
                foos[idx] = nullptr;
            }
            else
            {
                //printf( "RBA to realloc idx: %d\n", idx );
                foos[idx] = allocator.allocate();
                foos[idx] -> a = idx;
                foos[idx] -> b = idx;
                foos[idx] -> c = ( char ) idx;
            }
        }
    }
}
