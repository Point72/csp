/*
 * Implementation of the C API time types.
 *
 * CSP timestamps are nanoseconds since the Unix epoch in UTC, matching csp::DateTime. Calendar
 * conversion uses integer civil-date arithmetic rather than the C library, so it is free of
 * timezone and locale influence and is valid across the whole int64 range.
 */

#include <csp/engine/c/CspTime.h>

#include <limits.h>

namespace
{

const int64_t NANOS_PER_DAY = CCSP_SECONDS_PER_DAY * CCSP_NANOSECONDS_PER_SECOND;

/* Days from 1970-01-01 to the given civil date. Valid for any year in int range.
 * Algorithm from Howard Hinnant's chrono-compatible civil calendar. */
int64_t daysFromCivil( int64_t y, unsigned m, unsigned d )
{
    y -= m <= 2;
    const int64_t era = ( y >= 0 ? y : y - 399 ) / 400;
    const unsigned yoe = static_cast<unsigned>( y - era * 400 );
    const unsigned doy = ( 153 * ( m + ( m > 2 ? -3 : 9 ) ) + 2 ) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>( doe ) - 719468;
}

void civilFromDays( int64_t z, int64_t & y, unsigned & m, unsigned & d )
{
    z += 719468;
    const int64_t era = ( z >= 0 ? z : z - 146096 ) / 146097;
    const unsigned doe = static_cast<unsigned>( z - era * 146097 );
    const unsigned yoe = ( doe - doe / 1460 + doe / 36524 - doe / 146096 ) / 365;
    y = static_cast<int64_t>( yoe ) + era * 400;
    const unsigned doy = doe - ( 365 * yoe + yoe / 4 - yoe / 100 );
    const unsigned mp = ( 5 * doy + 2 ) / 153;
    d = doy - ( 153 * mp + 2 ) / 5 + 1;
    m = mp + ( mp < 10 ? 3 : -9 );
    y += ( m <= 2 );
}

bool isValidCivil( int year, int month, int day )
{
    if( month < 1 || month > 12 || day < 1 )
        return false;

    static const int lengths[ 12 ] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    int maxDay = lengths[ month - 1 ];
    if( month == 2 )
    {
        const bool leap = ( year % 4 == 0 && year % 100 != 0 ) || year % 400 == 0;
        if( leap )
            maxDay = 29;
    }
    return day <= maxDay;
}

/* Multiplies without wrapping; saturates at the int64 bounds */
int64_t saturatingMul( int64_t value, int64_t scale )
{
    if( value > INT64_MAX / scale )
        return INT64_MAX;
    if( value < INT64_MIN / scale )
        return INT64_MIN;
    return value * scale;
}

int64_t saturatingAdd( int64_t a, int64_t b )
{
    if( b > 0 && a > INT64_MAX - b )
        return INT64_MAX;
    if( b < 0 && a < INT64_MIN - b )
        return INT64_MIN;
    return a + b;
}

/* Converts a double count of units into nanoseconds, saturating out-of-range results */
int64_t nanosFromDouble( double units, double nanosPerUnit )
{
    const double nanos = units * nanosPerUnit;
    if( !( nanos >= -9223372036854775808.0 ) )
        return INT64_MIN;
    if( nanos >= 9223372036854775808.0 )
        return INT64_MAX;
    return static_cast<int64_t>( nanos );
}

}

extern "C" {

// ============================================================================
// DateTime
// ============================================================================

CCspDateTime ccsp_datetime_from_nanoseconds( int64_t nanoseconds )
{
    return nanoseconds;
}

CCspDateTime ccsp_datetime_from_seconds( int64_t seconds )
{
    return saturatingMul( seconds, CCSP_NANOSECONDS_PER_SECOND );
}

CCspDateTime ccsp_datetime_from_milliseconds( int64_t milliseconds )
{
    return saturatingMul( milliseconds, CCSP_NANOSECONDS_PER_MILLISECOND );
}

CCspDateTime ccsp_datetime_from_parts( int year, int month, int day,
                                       int hour, int minute, int second,
                                       int nanosecond )
{
    if( !isValidCivil( year, month, day ) )
        return CCSP_DATETIME_NONE;
    if( hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 )
        return CCSP_DATETIME_NONE;
    if( nanosecond < 0 || nanosecond >= CCSP_NANOSECONDS_PER_SECOND )
        return CCSP_DATETIME_NONE;

    const int64_t days = daysFromCivil( year, static_cast<unsigned>( month ), static_cast<unsigned>( day ) );
    const int64_t secondsOfDay = static_cast<int64_t>( hour ) * 3600 + static_cast<int64_t>( minute ) * 60 + second;

    int64_t result = saturatingMul( days, NANOS_PER_DAY );
    result = saturatingAdd( result, saturatingMul( secondsOfDay, CCSP_NANOSECONDS_PER_SECOND ) );
    result = saturatingAdd( result, nanosecond );
    return result;
}

int64_t ccsp_datetime_to_nanoseconds( CCspDateTime dt )
{
    return dt;
}

int64_t ccsp_datetime_to_seconds( CCspDateTime dt )
{
    return dt / CCSP_NANOSECONDS_PER_SECOND;
}

int64_t ccsp_datetime_to_milliseconds( CCspDateTime dt )
{
    return dt / CCSP_NANOSECONDS_PER_MILLISECOND;
}

void ccsp_datetime_to_parts( CCspDateTime dt,
                             int * out_year, int * out_month, int * out_day,
                             int * out_hour, int * out_minute, int * out_second,
                             int * out_nanosecond )
{
    if( dt == CCSP_DATETIME_NONE )
        return;

    int64_t days = dt / NANOS_PER_DAY;
    int64_t rem = dt % NANOS_PER_DAY;
    if( rem < 0 )
    {
        rem += NANOS_PER_DAY;
        --days;
    }

    int64_t year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civilFromDays( days, year, month, day );

    const int64_t secondsOfDay = rem / CCSP_NANOSECONDS_PER_SECOND;

    if( out_year )       *out_year = static_cast<int>( year );
    if( out_month )      *out_month = static_cast<int>( month );
    if( out_day )        *out_day = static_cast<int>( day );
    if( out_hour )       *out_hour = static_cast<int>( secondsOfDay / 3600 );
    if( out_minute )     *out_minute = static_cast<int>( ( secondsOfDay % 3600 ) / 60 );
    if( out_second )     *out_second = static_cast<int>( secondsOfDay % 60 );
    if( out_nanosecond ) *out_nanosecond = static_cast<int>( rem % CCSP_NANOSECONDS_PER_SECOND );
}

CCspDateTime ccsp_datetime_add( CCspDateTime dt, CCspTimeDelta delta )
{
    if( dt == CCSP_DATETIME_NONE )
        return CCSP_DATETIME_NONE;
    return saturatingAdd( dt, delta );
}

CCspTimeDelta ccsp_datetime_diff( CCspDateTime a, CCspDateTime b )
{
    if( a == CCSP_DATETIME_NONE || b == CCSP_DATETIME_NONE )
        return CCSP_TIMEDELTA_ZERO;
    return saturatingAdd( a, -b );
}

// ============================================================================
// TimeDelta
// ============================================================================

CCspTimeDelta ccsp_timedelta_from_nanoseconds( int64_t nanoseconds )
{
    return nanoseconds;
}

CCspTimeDelta ccsp_timedelta_from_microseconds( int64_t microseconds )
{
    return saturatingMul( microseconds, CCSP_NANOSECONDS_PER_MICROSECOND );
}

CCspTimeDelta ccsp_timedelta_from_milliseconds( int64_t milliseconds )
{
    return saturatingMul( milliseconds, CCSP_NANOSECONDS_PER_MILLISECOND );
}

CCspTimeDelta ccsp_timedelta_from_seconds( double seconds )
{
    return nanosFromDouble( seconds, 1e9 );
}

CCspTimeDelta ccsp_timedelta_from_minutes( double minutes )
{
    return nanosFromDouble( minutes, 60.0 * 1e9 );
}

CCspTimeDelta ccsp_timedelta_from_hours( double hours )
{
    return nanosFromDouble( hours, 3600.0 * 1e9 );
}

CCspTimeDelta ccsp_timedelta_from_days( double days )
{
    return nanosFromDouble( days, 86400.0 * 1e9 );
}

double ccsp_timedelta_to_seconds( CCspTimeDelta td )
{
    return static_cast<double>( td ) / 1e9;
}

int64_t ccsp_timedelta_to_nanoseconds( CCspTimeDelta td )
{
    return td;
}

// ============================================================================
// Date
// ============================================================================

CCspDate ccsp_date_from_days( int32_t days_since_epoch )
{
    return days_since_epoch;
}

CCspDate ccsp_date_from_parts( int year, int month, int day )
{
    if( !isValidCivil( year, month, day ) )
        return CCSP_DATE_NONE;

    const int64_t days = daysFromCivil( year, static_cast<unsigned>( month ), static_cast<unsigned>( day ) );
    if( days < INT32_MIN || days > INT32_MAX )
        return CCSP_DATE_NONE;
    return static_cast<CCspDate>( days );
}

int32_t ccsp_date_to_days( CCspDate date )
{
    return date;
}

void ccsp_date_to_parts( CCspDate date, int * out_year, int * out_month, int * out_day )
{
    if( date == CCSP_DATE_NONE )
        return;

    int64_t year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civilFromDays( date, year, month, day );

    if( out_year )  *out_year = static_cast<int>( year );
    if( out_month ) *out_month = static_cast<int>( month );
    if( out_day )   *out_day = static_cast<int>( day );
}

// ============================================================================
// Time of day
// ============================================================================

CCspTime ccsp_time_from_nanoseconds( int64_t nanoseconds_since_midnight )
{
    if( nanoseconds_since_midnight < 0 || nanoseconds_since_midnight >= NANOS_PER_DAY )
        return CCSP_TIME_NONE;
    return nanoseconds_since_midnight;
}

CCspTime ccsp_time_from_parts( int hour, int minute, int second, int nanosecond )
{
    if( hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 59 )
        return CCSP_TIME_NONE;
    if( nanosecond < 0 || nanosecond >= CCSP_NANOSECONDS_PER_SECOND )
        return CCSP_TIME_NONE;

    return ( static_cast<int64_t>( hour ) * 3600 + static_cast<int64_t>( minute ) * 60 + second )
           * CCSP_NANOSECONDS_PER_SECOND + nanosecond;
}

int64_t ccsp_time_to_nanoseconds( CCspTime time )
{
    return time;
}

void ccsp_time_to_parts( CCspTime time, int * out_hour, int * out_minute, int * out_second, int * out_nanosecond )
{
    if( time == CCSP_TIME_NONE )
        return;

    const int64_t secondsOfDay = time / CCSP_NANOSECONDS_PER_SECOND;

    if( out_hour )       *out_hour = static_cast<int>( secondsOfDay / 3600 );
    if( out_minute )     *out_minute = static_cast<int>( ( secondsOfDay % 3600 ) / 60 );
    if( out_second )     *out_second = static_cast<int>( secondsOfDay % 60 );
    if( out_nanosecond ) *out_nanosecond = static_cast<int>( time % CCSP_NANOSECONDS_PER_SECOND );
}

}
