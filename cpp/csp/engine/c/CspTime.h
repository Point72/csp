/*
 * ABI-stable C Time Types for CSP Engine
 *
 * CSP uses nanosecond precision for all time types internally.
 * These types map directly to the C++ csp::DateTime, csp::TimeDelta, etc.
 */
#ifndef _IN_CSP_ENGINE_C_CSPTIME_H
#define _IN_CSP_ENGINE_C_CSPTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DateTime: nanoseconds since Unix epoch (1970-01-01 00:00:00 UTC)
 * Maps to csp::DateTime
 */
typedef int64_t CCspDateTime;

/*
 * TimeDelta: duration in nanoseconds (can be negative)
 * Maps to csp::TimeDelta
 */
typedef int64_t CCspTimeDelta;

/*
 * Date: days since Unix epoch (1970-01-01)
 * Maps to csp::Date
 */
typedef int32_t CCspDate;

/*
 * Time: nanoseconds since midnight
 * Maps to csp::Time
 */
typedef int64_t CCspTime;

/* Constants */
#define CCSP_NANOSECONDS_PER_SECOND      1000000000LL
#define CCSP_NANOSECONDS_PER_MILLISECOND 1000000LL
#define CCSP_NANOSECONDS_PER_MICROSECOND 1000LL
#define CCSP_SECONDS_PER_DAY             86400LL

/* Special values */
#define CCSP_DATETIME_MIN INT64_MIN
#define CCSP_DATETIME_MAX INT64_MAX
#define CCSP_TIMEDELTA_ZERO 0LL

/* DateTime construction */
CCspDateTime ccsp_datetime_from_nanoseconds( int64_t nanoseconds );
CCspDateTime ccsp_datetime_from_seconds( int64_t seconds );
CCspDateTime ccsp_datetime_from_milliseconds( int64_t milliseconds );
CCspDateTime ccsp_datetime_from_parts(
    int year, int month, int day,
    int hour, int minute, int second,
    int nanosecond
);

/* DateTime extraction */
int64_t ccsp_datetime_to_nanoseconds( CCspDateTime dt );
int64_t ccsp_datetime_to_seconds( CCspDateTime dt );
int64_t ccsp_datetime_to_milliseconds( CCspDateTime dt );
void ccsp_datetime_to_parts(
    CCspDateTime dt,
    int* out_year, int* out_month, int* out_day,
    int* out_hour, int* out_minute, int* out_second,
    int* out_nanosecond
);

/* DateTime arithmetic */
CCspDateTime ccsp_datetime_add( CCspDateTime dt, CCspTimeDelta delta );
CCspTimeDelta ccsp_datetime_diff( CCspDateTime a, CCspDateTime b );

/* TimeDelta construction */
CCspTimeDelta ccsp_timedelta_from_nanoseconds( int64_t nanoseconds );
CCspTimeDelta ccsp_timedelta_from_microseconds( int64_t microseconds );
CCspTimeDelta ccsp_timedelta_from_milliseconds( int64_t milliseconds );
CCspTimeDelta ccsp_timedelta_from_seconds( double seconds );
CCspTimeDelta ccsp_timedelta_from_minutes( double minutes );
CCspTimeDelta ccsp_timedelta_from_hours( double hours );
CCspTimeDelta ccsp_timedelta_from_days( double days );

/* TimeDelta extraction */
double ccsp_timedelta_to_seconds( CCspTimeDelta td );
int64_t ccsp_timedelta_to_nanoseconds( CCspTimeDelta td );

/* Date construction */
CCspDate ccsp_date_from_days( int32_t days_since_epoch );
CCspDate ccsp_date_from_parts( int year, int month, int day );

/* Date extraction */
int32_t ccsp_date_to_days( CCspDate date );
void ccsp_date_to_parts( CCspDate date, int * out_year, int * out_month, int * out_day );

/* Time (time of day) construction */
CCspTime ccsp_time_from_nanoseconds( int64_t nanoseconds_since_midnight );
CCspTime ccsp_time_from_parts( int hour, int minute, int second, int nanosecond );

/* Time extraction */
int64_t ccsp_time_to_nanoseconds( CCspTime time );
void ccsp_time_to_parts( CCspTime time, int * out_hour, int * out_minute, int * out_second, int * out_nanosecond );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPTIME_H */
