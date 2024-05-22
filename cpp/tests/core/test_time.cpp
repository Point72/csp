#include <gtest/gtest.h>
#include <csp/core/Time.h>
#include <thread>

using namespace csp;

TEST( DateTimeTest, test_basic_functionality )
{
    DateTime t1 = DateTime::now();
    std::this_thread::sleep_for( std::chrono::milliseconds(50) );
    EXPECT_NE( t1, DateTime::now() );

    DateTime d1( 2017, 1, 1, 13, 1, 17, 123456789 );
    DateTime d2( 2017, 1, 1, 13, 1, 17, 123456788 );

    EXPECT_EQ( d1 - d2, TimeDelta::fromNanoseconds( 1 ) );
    EXPECT_EQ( d2 - d1, TimeDelta::fromNanoseconds( -1 ) );

    EXPECT_TRUE( d1 > d2 );
    EXPECT_TRUE( d1 >= d2 );
    EXPECT_FALSE( d1 < d2 );
    EXPECT_FALSE( d1 < d2 );

    EXPECT_TRUE( d1 - TimeDelta::fromNanoseconds( 1 ) <= d2 );
    EXPECT_FALSE( d1 - TimeDelta::fromNanoseconds( 1 ) < d2 );
    
    d1 = DateTime( 2017, 1, 1, 1, 1 );
    d2 = DateTime( 2017, 1, 2, 2, 2 );

    TimeDelta delta = d2 - d1;
    EXPECT_EQ( delta.days(), 1 );
    EXPECT_EQ( delta.hours(), 1 );
    EXPECT_EQ( delta.minutes(), 1 );
    EXPECT_EQ( delta.seconds(), 0 );
    EXPECT_EQ( delta.nanoseconds(), 0 );

    EXPECT_EQ( d1 + delta, d2 );
    EXPECT_EQ( d2 - delta, d1 );

    EXPECT_EQ( DateTime( Date( 2017, 1, 2 ), Time( 15, 14, 14, 123456 ) ), DateTime( 2017, 1, 2, 15, 14, 14, 123456 ) );

    EXPECT_EQ( DateTime( 2017, 1, 1, 1, 1, 1, 123456789 ).roundDown( TimeDelta::fromSeconds( 1 ) ), DateTime( 2017, 1, 1, 1, 1, 1 ) );

    EXPECT_EQ( DateTime( 2017, 1, 1, 1, 1, 1, 823456789 ).roundDown( TimeDelta::fromMilliseconds( 500 ) ), DateTime( 2017, 1, 1, 1, 1, 1, 500000000 ) );
    EXPECT_EQ( DateTime( 2017, 1, 1, 1, 1, 1, 123456789 ).roundDown( TimeDelta::fromMilliseconds( 500 ) ), DateTime( 2017, 1, 1, 1, 1, 1 ) );
    EXPECT_EQ( DateTime( 2017, 1, 1, 1, 1, 1, 123456789 ).roundDown( TimeDelta::fromSeconds( 30 ) ), DateTime( 2017, 1, 1, 1, 1, 0 ) );
    EXPECT_EQ( DateTime( 2017, 1, 1, 1, 1, 31, 123456789 ).roundDown( TimeDelta::fromSeconds( 30 ) ), DateTime( 2017, 1, 1, 1, 1, 30 ) );

    ASSERT_EQ( DateTime( 2018, 1, 1, 14, 1, 2, 123456789 ).asString(), "20180101 14:01:02.123456789" );

    ASSERT_EQ( DateTime::MIN_VALUE().asString(), "min" );
    ASSERT_EQ( DateTime::MAX_VALUE().asString(), "max" );
    ASSERT_EQ( DateTime::NONE().asString(),     "none" );

    ASSERT_TRUE( DateTime().isNone() );
}
    
TEST( TimeDeltaTest, test_basic_functionality )
{
    TimeDelta delta1 = TimeDelta::fromSeconds( 86400 * 3 + 3600 * 4 + 5 * 60 + 15 );
    EXPECT_EQ( delta1.days(), 3 );
    EXPECT_EQ( delta1.hours(), 4 );
    EXPECT_EQ( delta1.minutes(), 5 );
    EXPECT_EQ( delta1.seconds(), 15 );
    EXPECT_EQ( delta1.nanoseconds(), 0 );

    TimeDelta delta2 = TimeDelta::fromNanoseconds( (86400 * 3 + 3600 * 4 + 5 * 60 + 15 ) * NANOS_PER_SECOND + 123456789 );
    EXPECT_EQ( delta2.days(), 3 );
    EXPECT_EQ( delta2.hours(), 4 );
    EXPECT_EQ( delta2.minutes(), 5 );
    EXPECT_EQ( delta2.seconds(), 15 );
    EXPECT_EQ( delta2.nanoseconds(), 123456789 );

    EXPECT_EQ( delta1, delta1 );
    EXPECT_NE( delta1, delta2 );

    EXPECT_EQ( TimeDelta( 5, 123 ) * 3, TimeDelta( 15, 369 ) );
    {
        TimeDelta d( 5, 123 );
        d *= 3;
        EXPECT_EQ( d, TimeDelta( 15, 369 ) );
    }

    EXPECT_EQ( TimeDelta( 15, 369 ) / 3, TimeDelta( 5, 123 ) );
    {
        TimeDelta d( 15, 369 );
        d /= 3;
        EXPECT_EQ( d, TimeDelta( 5, 123 ) );
    }

    EXPECT_EQ( TimeDelta( 15, 0 ) / TimeDelta( 5, 0 ), 3 );
    EXPECT_EQ( TimeDelta( 2, 0 ) / TimeDelta( -1, 0 ), -2 );

    ASSERT_EQ( delta1.asString(), "3 days 04:05:15" );
    ASSERT_EQ( delta2.asString(), "3 days 04:05:15.123456789" );

    ASSERT_EQ( TimeDelta::fromString( "04:05:15" ),  TimeDelta::fromSeconds( 4 * 3600 + 5 * 60 + 15 ) );
    ASSERT_EQ( TimeDelta::fromString( "05:15.123" ), TimeDelta::fromSeconds( 5 * 60 + 15 ) + TimeDelta::fromMilliseconds( 123 ) );
    ASSERT_EQ( TimeDelta::fromString( "2.123456" ),  TimeDelta::fromSeconds( 2 ) + TimeDelta::fromMicroseconds( 123456 ) );

    ASSERT_EQ( TimeDelta::fromSeconds( 3 * 60 + 15 ).asString(), "00:03:15" );    
    ASSERT_EQ( TimeDelta::fromHours( 25 ).asString(), "1 day 01:00:00" );    
    ASSERT_EQ( TimeDelta::fromDays( 25 ).asString(), "25 days 00:00:00" );    

    {
        TimeDelta t1 = TimeDelta::fromSeconds( 60 );
        TimeDelta t2 = TimeDelta::fromSeconds( 5 );

        ASSERT_EQ( t1 + t2, TimeDelta::fromSeconds( 65 ) );
        ASSERT_EQ( t1 - t2, TimeDelta::fromSeconds( 55 ) );
        ASSERT_EQ( t2 - t1, TimeDelta::fromSeconds( -55 ) );

        t1 += t2;
        ASSERT_EQ( t1, TimeDelta::fromSeconds( 65 ) );
        t1 -= t2;
        ASSERT_EQ( t1, TimeDelta::fromSeconds( 60 ) );
    }

    {
        TimeDelta td = TimeDelta::fromMicroseconds( 123 );
        td = -td;
        ASSERT_EQ( td, TimeDelta::fromMicroseconds( -123 ) );
        td = -td;
        ASSERT_EQ( td, TimeDelta::fromMicroseconds( 123 ) );
    }

    ASSERT_TRUE( TimeDelta().isNone() );
}

TEST( DateTimeEx, test_basic_functionality )
{
    DateTime dt( 2017, 3, 7, 13, 2, 17, 123456789 );
    DateTimeEx ex( dt );

    ASSERT_EQ( ex.year(),   2017 );
    ASSERT_EQ( ex.month(),  3 );
    ASSERT_EQ( ex.day(),    7 );
    ASSERT_EQ( ex.hour(),   13 );
    ASSERT_EQ( ex.minute(), 2 );
    ASSERT_EQ( ex.second(), 17 );

    ASSERT_EQ( ex.milliseconds(), 123 );
    ASSERT_EQ( ex.microseconds(), 123456 );
    ASSERT_EQ( ex.nanoseconds(),  123456789 );


    //negative ticks, pre-1970
    dt = DateTime( 1969, 12, 31, 23, 59, 59, 123456789 );
    DateTimeEx ex2( dt );

    ASSERT_EQ( ex2.year(),   1969 );
    ASSERT_EQ( ex2.month(),  12 );
    ASSERT_EQ( ex2.day(),    31 );
    ASSERT_EQ( ex2.hour(),   23 );
    ASSERT_EQ( ex2.minute(), 59 );
    ASSERT_EQ( ex2.second(), 59 );

    ASSERT_EQ( ex2.milliseconds(), 123 );
    ASSERT_EQ( ex2.microseconds(), 123456 );
    ASSERT_EQ( ex2.nanoseconds(),  123456789 );

    //Only linux supports time before EPOCH
#ifdef __linux__
    dt = DateTime( 1888, 11, 15, 23, 15, 59, 999999999 );
    DateTimeEx ex3( dt );

    ASSERT_EQ( ex3.year(),   1888 );
    ASSERT_EQ( ex3.month(),  11 );
    ASSERT_EQ( ex3.day(),    15 );
    ASSERT_EQ( ex3.hour(),   23 );
    ASSERT_EQ( ex3.minute(), 15 );
    ASSERT_EQ( ex3.second(), 59 );

    ASSERT_EQ( ex3.milliseconds(), 999 );
    ASSERT_EQ( ex3.microseconds(), 999999 );
    ASSERT_EQ( ex3.nanoseconds(),  999999999 );
#endif
}

TEST( TimeTest, test_basic_functionality )
{
    Time t( 14, 1, 2, 123456789 );
    ASSERT_EQ( t.hour(),       14 );
    ASSERT_EQ( t.minute(),     1 );
    ASSERT_EQ( t.second(),     2 );
    ASSERT_EQ( t.nanosecond(), 123456789 );

    Time t2( 14, 1, 2, 123456790 );
    ASSERT_TRUE( t2 > t );
    ASSERT_TRUE( t < t2 );
    ASSERT_FALSE( t == t2 );
    ASSERT_TRUE( t != t2 );
    
    ASSERT_EQ( t.asString(), "14:01:02.123456789" );
    ASSERT_EQ( Time::fromString( "14:01:02.123456789" ), t );
    ASSERT_EQ( Time::fromString( "14:01:02.1234" ), Time( 14, 1, 2, 123400000 ) );
    ASSERT_EQ( Time::fromString( "14:01:02.12" ),   Time( 14, 1, 2, 120000000 ) );
    ASSERT_EQ( Time::fromString( "14:01:02" ),      Time( 14, 1, 2 ) );
    ASSERT_EQ( Time::fromString( "14:01" ),         Time( 14, 1, 0 ) );
    ASSERT_EQ( Time::fromString( "14" ),            Time( 14, 0, 0 ) );

    DateTime dt( 2017, 1, 1, 14, 1, 2, 123456789 );
    ASSERT_EQ( dt.time(), t );

    DateTime dt2( 2017, 1, 1, 1, 1, 1 );
    DateTime dt3 = dt2.withTime( t );

    ASSERT_EQ( dt3, DateTime( 2017, 1, 1, 14, 1, 2, 123456789 ) );

    //math
    Time t3( 14, 30, 0 );
    ASSERT_EQ( t3 + TimeDelta::fromSeconds( 91 ), Time( 14, 31, 31 ) );
    ASSERT_EQ( t3 - TimeDelta::fromSeconds( 91 ), Time( 14, 28, 29 ) );

    t3 += TimeDelta::fromSeconds( 91 );
    ASSERT_EQ( t3, Time( 14, 31, 31 ) );
    t3 -= TimeDelta::fromSeconds( 91 );
    ASSERT_EQ( t3, Time( 14, 30, 0 ) );

    ASSERT_EQ( Time( 14, 31, 31 ) - Time( 14, 30, 0 ), TimeDelta::fromSeconds( 91 ) );

    //"daylight savings" ( not that there is any, its UTC
    dt2 = DateTime( 2017, 5, 1, 1, 1, 1);
    dt3 = dt2.withTime( t );
    ASSERT_EQ( dt3, DateTime( 2017, 5, 1, 14, 1, 2, 123456789 ) );

    ASSERT_TRUE( Time().isNone() );
}

TEST( DateTest, test_strfortime )
{
    Date d( 2017, 2, 3 );
    std::string format = "/tmp/univ_%Y%m%d.txt";
    char fileName[1024];
    d.strftime( fileName, sizeof( fileName ), format.c_str());
    ASSERT_STREQ( "/tmp/univ_20170203.txt", fileName );

    format = "/data_root/univ/%Y/%m";
    d.strftime( fileName, sizeof( fileName ), format.c_str());
    ASSERT_STREQ( "/data_root/univ/2017/02", fileName );

    ASSERT_TRUE( Date().isNone() );
}

TEST( DateTest, test_basic_functionality )
{
    Date d( 2017, 2, 3 );
    ASSERT_EQ( d.year(),  2017 );
    ASSERT_EQ( d.month(), 2 );
    ASSERT_EQ( d.day(),   3 );

    ASSERT_EQ( d, Date( 2017, 2, 3 ) );
    ASSERT_NE( d, Date( 2017, 2, 4 ) );
    ASSERT_TRUE( d < Date( 2017, 2, 4 ) );
    ASSERT_TRUE( d > Date( 2017, 2, 2 ) );

    ASSERT_TRUE( d < Date( 2017, 3, 3 ) );
    ASSERT_TRUE( d > Date( 2017, 1, 3 ) );

    ASSERT_TRUE( d < Date( 2018, 2, 3 ) );
    ASSERT_TRUE( d < Date( 2018, 1, 1 ) );
    ASSERT_TRUE( d > Date( 2016, 2, 3 ) );
    ASSERT_TRUE( d > Date( 2016, 5, 5 ) );

    ASSERT_TRUE( Date( 2017, 9, 20 ) < Date( 2017, 10, 17 ) );

    ASSERT_EQ( d.asString(), "20170203" );
    ASSERT_EQ( Date::fromYYYYMMDD( "20170203" ), d );

    DateTime dt( 2017, 2, 3, 15, 16, 7 );
    ASSERT_EQ( dt.date(), d );

    //date math tests
    Date d1( 2017, 1, 1 );
    Date d2( 2017, 2, 1 );
    ASSERT_EQ( d2 - d1, TimeDelta::fromDays( 31 ) );
    ASSERT_EQ( d1 - d2, TimeDelta::fromDays( -31 ) );
    ASSERT_EQ( d1 + TimeDelta::fromDays( 31 ), d2 );
    ASSERT_EQ( d2 - TimeDelta::fromDays( 31 ), d1 );
    
    Date d3 = d1;
    d3 += TimeDelta::fromDays( 31 );
    ASSERT_EQ( d3, d2 );
    d3 -= TimeDelta::fromDays( 31 );
    ASSERT_EQ( d3, d1 );

    //weekday
    {
        Date base( 2017, 1, 1 );
        for( int days = 0; days < 7; ++days )
        {
            Date d = base + TimeDelta::fromDays( days );
            ASSERT_EQ( d.weekday(), days ) << d;
            ASSERT_EQ( d.isWeekend(), ( days == 0 || days == 6 ) ) << d;
        }
    }
}

TEST( sleep, basic_functionality )
{
    TimeDelta waittime = TimeDelta::fromSeconds( 1 );
    DateTime t1 = DateTime::now();

    csp::sleep( waittime );
    DateTime t2 = DateTime::now();

    ASSERT_GT( t2 - t1, waittime );
}

