#include <csp/engine/WindowBuffer.h>
#include <gtest/gtest.h>

//Note clear why windows linker is looknig for this
#ifdef WIN32
bool csp::DialectGenericType::operator==( struct csp::DialectGenericType const & ) const { return false; }
#endif

TEST( WindowBufferTest, test_time_window )
{
    csp::VariableSizeWindowBuffer<double> buffer;
    for (int i = 1; i <= 10; i++) buffer.push(i);
    for (int i = 1; i <= 8; i++) buffer.pop_left();
    for (int i = 11; i <= 18; i++) buffer.push(i);
    for (int i = 1; i <= 3; i++) buffer.pop_right();
    
    ASSERT_EQ( buffer.capacity(), 16 ); // 7 values in buffer but does not downsize
    ASSERT_EQ( buffer[0], 15);
    ASSERT_EQ( buffer[6], 9); // 6 ticks away
    ASSERT_EQ( buffer.count(), 7 );
}

TEST( WindowBufferTest, test_tick_window )
{
    csp::FixedSizeWindowBuffer<double> buffer(8);
    std::vector<int> rem;
    for (int i = 1; i <= 10; i++) 
    {
        if (buffer.full() && rem.size() < 8)
            rem.push_back(buffer.pop_left());
        buffer.push(i);
    }
    ASSERT_EQ( buffer.count(), 8 );
    ASSERT_EQ( rem.size(), 2 );
    ASSERT_EQ( rem[0], 1 );
    ASSERT_EQ( rem[1], 2 );

    for (int i = 1; i <= 5; i++) buffer.pop_left();
    ASSERT_EQ( buffer.count(), 3 );

    for (int i = 11; i <= 25; i++) 
    {
        if (buffer.full() && rem.size() < 8)
            rem.push_back(buffer.pop_left());
        buffer.push(i);
    }
    ASSERT_EQ( buffer.capacity(), 8 ); // 8 values in buffer
    ASSERT_EQ( buffer[0], 25);
    ASSERT_EQ( buffer[7], 18); // 7 ticks away
    ASSERT_EQ( buffer.count(), 8 );

    // Removals is full, discards unimportant values
    ASSERT_EQ( rem.size(), 8 );
    ASSERT_EQ( rem[0], 1 );
    ASSERT_EQ( rem[7], 13 );
}


