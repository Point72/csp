#include <gtest/gtest.h>

#include <csp/engine/c/CspAbi.h>
#include <csp/engine/c/CspError.h>
#include <csp/engine/c/CspString.h>
#include <csp/engine/c/CspTime.h>
#include <csp/engine/c/CspValue.h>
#include <csp/engine/c/AdapterManager.h>
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/OutputAdapter.h>

#include <string>

// ============================================================================
// Time
// ============================================================================

TEST( CApiTime, DateTimeRoundTripsThroughParts )
{
    const CCspDateTime dt = ccsp_datetime_from_parts( 2024, 2, 29, 13, 45, 7, 123456789 );
    ASSERT_NE( dt, CCSP_DATETIME_NONE );

    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0, ns = 0;
    ccsp_datetime_to_parts( dt, &y, &mo, &d, &h, &mi, &s, &ns );

    EXPECT_EQ( y, 2024 );
    EXPECT_EQ( mo, 2 );
    EXPECT_EQ( d, 29 );
    EXPECT_EQ( h, 13 );
    EXPECT_EQ( mi, 45 );
    EXPECT_EQ( s, 7 );
    EXPECT_EQ( ns, 123456789 );
}

TEST( CApiTime, EpochIsZeroAndDistinctFromNone )
{
    EXPECT_EQ( ccsp_datetime_from_parts( 1970, 1, 1, 0, 0, 0, 0 ), 0 );
    EXPECT_NE( CCSP_DATETIME_NONE, 0 );
}

TEST( CApiTime, PreEpochDatesDecomposeCorrectly )
{
    const CCspDateTime dt = ccsp_datetime_from_parts( 1969, 12, 31, 23, 59, 59, 999999999 );
    ASSERT_NE( dt, CCSP_DATETIME_NONE );
    EXPECT_EQ( dt, -1 );

    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0, ns = 0;
    ccsp_datetime_to_parts( dt, &y, &mo, &d, &h, &mi, &s, &ns );
    EXPECT_EQ( y, 1969 );
    EXPECT_EQ( mo, 12 );
    EXPECT_EQ( d, 31 );
    EXPECT_EQ( h, 23 );
    EXPECT_EQ( ns, 999999999 );
}

TEST( CApiTime, InvalidPartsAreRejected )
{
    EXPECT_EQ( ccsp_datetime_from_parts( 2023, 2, 29, 0, 0, 0, 0 ), CCSP_DATETIME_NONE );
    EXPECT_EQ( ccsp_datetime_from_parts( 2024, 13, 1, 0, 0, 0, 0 ), CCSP_DATETIME_NONE );
    EXPECT_EQ( ccsp_datetime_from_parts( 2024, 1, 1, 24, 0, 0, 0 ), CCSP_DATETIME_NONE );
    EXPECT_EQ( ccsp_datetime_from_parts( 2024, 1, 1, 0, 0, 0, 1000000000 ), CCSP_DATETIME_NONE );
}

TEST( CApiTime, DateRoundTrips )
{
    const CCspDate date = ccsp_date_from_parts( 2000, 3, 1 );
    ASSERT_NE( date, CCSP_DATE_NONE );

    int y = 0, m = 0, d = 0;
    ccsp_date_to_parts( date, &y, &m, &d );
    EXPECT_EQ( y, 2000 );
    EXPECT_EQ( m, 3 );
    EXPECT_EQ( d, 1 );

    EXPECT_EQ( ccsp_date_from_parts( 1970, 1, 1 ), 0 );
}

TEST( CApiTime, TimeOfDayRoundTripsAndRangeChecks )
{
    const CCspTime t = ccsp_time_from_parts( 23, 59, 59, 999999999 );
    ASSERT_NE( t, CCSP_TIME_NONE );

    int h = 0, m = 0, s = 0, ns = 0;
    ccsp_time_to_parts( t, &h, &m, &s, &ns );
    EXPECT_EQ( h, 23 );
    EXPECT_EQ( m, 59 );
    EXPECT_EQ( s, 59 );
    EXPECT_EQ( ns, 999999999 );

    EXPECT_EQ( ccsp_time_from_parts( 24, 0, 0, 0 ), CCSP_TIME_NONE );
    EXPECT_EQ( ccsp_time_from_nanoseconds( -1 ), CCSP_TIME_NONE );
}

TEST( CApiTime, TimeDeltaConversions )
{
    EXPECT_EQ( ccsp_timedelta_from_seconds( 1.5 ), 1500000000LL );
    EXPECT_EQ( ccsp_timedelta_from_minutes( 2.0 ), 120000000000LL );
    EXPECT_EQ( ccsp_timedelta_from_hours( 1.0 ), 3600000000000LL );
    EXPECT_EQ( ccsp_timedelta_from_days( 1.0 ), 86400000000000LL );
    EXPECT_DOUBLE_EQ( ccsp_timedelta_to_seconds( 1500000000LL ), 1.5 );
}

TEST( CApiTime, ArithmeticSaturatesInsteadOfWrapping )
{
    EXPECT_EQ( ccsp_datetime_add( CCSP_DATETIME_MAX, 1 ), CCSP_DATETIME_MAX );
    EXPECT_EQ( ccsp_datetime_from_seconds( INT64_MAX ), INT64_MAX );
    EXPECT_EQ( ccsp_datetime_add( CCSP_DATETIME_NONE, 5 ), CCSP_DATETIME_NONE );
}

// ============================================================================
// String
// ============================================================================

TEST( CApiString, OwnedStringCopiesAndFrees )
{
    CCspString s = ccsp_string_create( "hello", 5 );
    ASSERT_NE( s.data, nullptr );
    EXPECT_EQ( s.length, 5u );
    EXPECT_STREQ( s.data, "hello" );

    CCspStringView view = ccsp_string_as_view( &s );
    EXPECT_EQ( view.length, 5u );

    ccsp_string_free( &s );
    EXPECT_EQ( s.data, nullptr );
    EXPECT_EQ( s.length, 0u );

    // free is idempotent
    ccsp_string_free( &s );
}

TEST( CApiString, HandlesEmbeddedNulls )
{
    const char raw[] = { 'a', '\0', 'b' };
    CCspString s = ccsp_string_create( raw, sizeof( raw ) );
    ASSERT_NE( s.data, nullptr );
    EXPECT_EQ( s.length, 3u );
    EXPECT_EQ( std::string( s.data, s.length ), std::string( raw, sizeof( raw ) ) );
    ccsp_string_free( &s );
}

TEST( CApiString, ViewsHandleNull )
{
    CCspStringView view = ccsp_string_view_from_cstr( nullptr );
    EXPECT_EQ( view.length, 0u );
    EXPECT_TRUE( ccsp_string_view_is_empty( view ) );
}

// ============================================================================
// Value
// ============================================================================

TEST( CApiValue, ScalarRoundTripAndTypeMismatch )
{
    CCspValue v;
    ccsp_value_init( &v );
    EXPECT_FALSE( ccsp_value_is_valid( &v ) );

    ccsp_value_set_int64( &v, 42 );
    EXPECT_TRUE( ccsp_value_is_valid( &v ) );
    EXPECT_TRUE( ccsp_value_is_integer( &v ) );
    EXPECT_TRUE( ccsp_value_is_numeric( &v ) );

    int64_t out = 0;
    EXPECT_EQ( ccsp_value_get_int64( &v, &out ), CCSP_OK );
    EXPECT_EQ( out, 42 );

    double dout = 0;
    EXPECT_EQ( ccsp_value_get_double( &v, &dout ), CCSP_ERROR_TYPE_MISMATCH );

    ccsp_value_free( &v );
}

TEST( CApiValue, OwnedStringIsCopiedAndReleased )
{
    CCspValue v;
    ccsp_value_init( &v );

    ASSERT_EQ( ccsp_value_set_string( &v, "abc", 3 ), CCSP_OK );

    const char * data = nullptr;
    size_t len = 0;
    ASSERT_EQ( ccsp_value_get_string( &v, &data, &len ), CCSP_OK );
    EXPECT_EQ( std::string( data, len ), "abc" );

    // Overwriting must release the previous allocation rather than leak it
    ccsp_value_set_int32( &v, 7 );
    EXPECT_TRUE( ccsp_value_is_type( &v, CCSP_TYPE_INT32 ) );

    ccsp_value_free( &v );
}

TEST( CApiValue, CopyDeepCopiesOwnedStrings )
{
    CCspValue src, dst;
    ccsp_value_init( &src );
    ccsp_value_init( &dst );

    ASSERT_EQ( ccsp_value_set_string( &src, "payload", 7 ), CCSP_OK );
    ASSERT_EQ( ccsp_value_copy( &dst, &src ), CCSP_OK );

    const char * srcData = nullptr;
    const char * dstData = nullptr;
    size_t srcLen = 0, dstLen = 0;
    ASSERT_EQ( ccsp_value_get_string( &src, &srcData, &srcLen ), CCSP_OK );
    ASSERT_EQ( ccsp_value_get_string( &dst, &dstData, &dstLen ), CCSP_OK );

    EXPECT_NE( srcData, dstData );
    EXPECT_EQ( std::string( dstData, dstLen ), "payload" );

    // Freeing the source must leave the copy intact
    ccsp_value_free( &src );
    ASSERT_EQ( ccsp_value_get_string( &dst, &dstData, &dstLen ), CCSP_OK );
    EXPECT_EQ( std::string( dstData, dstLen ), "payload" );

    ccsp_value_free( &dst );
}

TEST( CApiValue, MoveTransfersOwnership )
{
    CCspValue src, dst;
    ccsp_value_init( &src );
    ccsp_value_init( &dst );

    ASSERT_EQ( ccsp_value_set_string( &src, "moved", 5 ), CCSP_OK );
    ccsp_value_move( &dst, &src );

    EXPECT_FALSE( ccsp_value_is_valid( &src ) );
    EXPECT_TRUE( ccsp_value_is_type( &dst, CCSP_TYPE_STRING ) );

    ccsp_value_free( &dst );
    ccsp_value_free( &src );
}

TEST( CApiValue, StringViewIsBorrowedNotFreed )
{
    const char * backing = "borrowed";

    CCspValue v;
    ccsp_value_init( &v );
    ccsp_value_set_string_view( &v, backing, 8 );

    const char * data = nullptr;
    size_t len = 0;
    ASSERT_EQ( ccsp_value_get_string( &v, &data, &len ), CCSP_OK );
    EXPECT_EQ( data, backing );

    // Must not attempt to free the caller's literal
    ccsp_value_free( &v );
}

TEST( CApiValue, NullArgumentsAreRejected )
{
    int64_t out = 0;
    EXPECT_EQ( ccsp_value_get_int64( nullptr, &out ), CCSP_ERROR_NULL_POINTER );

    CCspValue v;
    ccsp_value_init( &v );
    ccsp_value_set_int64( &v, 1 );
    EXPECT_EQ( ccsp_value_get_int64( &v, nullptr ), CCSP_ERROR_NULL_POINTER );
    ccsp_value_free( &v );
}

// ============================================================================
// ABI versioning
// ============================================================================

namespace
{
/* A table as it would look if built against an older header with fewer callbacks */
struct OldVTable
{
    uint32_t abi_version;
    uint32_t struct_size;
    void * user_data;
};

void destroyCb( void * ) {}
}

TEST( CApiAbi, CurrentTableRoundTrips )
{
    CCspPushInputAdapterVTable src, dst;
    CCSP_VTABLE_INIT( &src, CCspPushInputAdapterVTable );
    src.user_data = reinterpret_cast<void *>( 0x1234 );
    src.destroy = destroyCb;

    ASSERT_EQ( ccsp_vtable_adopt( &dst, sizeof( dst ), &src ), 1 );
    EXPECT_EQ( dst.user_data, reinterpret_cast<void *>( 0x1234 ) );
    EXPECT_EQ( dst.destroy, destroyCb );
    EXPECT_EQ( dst.struct_size, static_cast<uint32_t>( sizeof( CCspPushInputAdapterVTable ) ) );
}

TEST( CApiAbi, OlderSmallerTableIsAcceptedAndZeroFilled )
{
    OldVTable old;
    old.abi_version = CCSP_ABI_VERSION;
    old.struct_size = static_cast<uint32_t>( sizeof( OldVTable ) );
    old.user_data = reinterpret_cast<void *>( 0xBEEF );

    CCspPushInputAdapterVTable dst;
    ASSERT_EQ( ccsp_vtable_adopt( &dst, sizeof( dst ), &old ), 1 );
    EXPECT_EQ( dst.user_data, reinterpret_cast<void *>( 0xBEEF ) );
    EXPECT_EQ( dst.destroy, nullptr );
    EXPECT_EQ( dst.start, nullptr );
}

TEST( CApiAbi, NewerOrUninitializedTablesAreRejected )
{
    CCspPushInputAdapterVTable src, dst;

    CCSP_VTABLE_INIT( &src, CCspPushInputAdapterVTable );
    src.abi_version = CCSP_ABI_VERSION + 1;
    EXPECT_EQ( ccsp_vtable_adopt( &dst, sizeof( dst ), &src ), 0 );

    memset( &src, 0, sizeof( src ) );
    EXPECT_EQ( ccsp_vtable_adopt( &dst, sizeof( dst ), &src ), 0 );

    EXPECT_EQ( ccsp_vtable_adopt( nullptr, sizeof( dst ), &src ), 0 );
    EXPECT_EQ( ccsp_vtable_adopt( &dst, sizeof( dst ), nullptr ), 0 );
}

TEST( CApiAbi, VTablesLeadWithTheAbiHeader )
{
    EXPECT_EQ( offsetof( CCspPushInputAdapterVTable, abi_version ), 0u );
    EXPECT_EQ( offsetof( CCspPushInputAdapterVTable, struct_size ), sizeof( uint32_t ) );
    EXPECT_EQ( offsetof( CCspOutputAdapterVTable, abi_version ), 0u );
    EXPECT_EQ( offsetof( CCspOutputAdapterVTable, struct_size ), sizeof( uint32_t ) );
    EXPECT_EQ( offsetof( CCspAdapterManagerVTable, abi_version ), 0u );
    EXPECT_EQ( offsetof( CCspAdapterManagerVTable, struct_size ), sizeof( uint32_t ) );
}

// ============================================================================
// Error state
// ============================================================================

TEST( CApiError, ErrorStateIsRecordedAndCleared )
{
    ccsp_clear_error();
    EXPECT_EQ( ccsp_get_last_error(), CCSP_OK );

    ccsp_set_error( CCSP_ERROR_TYPE_MISMATCH, "boom" );
    EXPECT_EQ( ccsp_get_last_error(), CCSP_ERROR_TYPE_MISMATCH );
    EXPECT_STREQ( ccsp_get_last_error_message(), "boom" );

    ccsp_clear_error();
    EXPECT_EQ( ccsp_get_last_error(), CCSP_OK );
    EXPECT_EQ( ccsp_get_last_error_message(), nullptr );
}
