/*
 * ABI-stable C Time Types for CSP Engine
 *
 * CSP uses nanosecond precision for all time types internally.
 * These types map directly to the C++ csp::DateTime, csp::TimeDelta, etc.
 */
#ifndef _IN_CSP_ENGINE_C_CSPTIME_H
#define _IN_CSP_ENGINE_C_CSPTIME_H

#include <csp/engine/c/CspExport.h>
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
/* Sentinel for "no datetime". Distinct from 0, which is the valid Unix epoch timestamp. */
#define CCSP_DATETIME_NONE CCSP_DATETIME_MIN
#define CCSP_TIMEDELTA_ZERO 0LL
/* Sentinels for the date and time-of-day types */
#define CCSP_DATE_NONE INT32_MIN
#define CCSP_TIME_NONE INT64_MIN

/* DateTime construction. All conversions are UTC. */
CSP_C_API_EXPORT CCspDateTime ccsp_datetime_from_nanoseconds( int64_t nanoseconds );
CSP_C_API_EXPORT CCspDateTime ccsp_datetime_from_seconds( int64_t seconds );
CSP_C_API_EXPORT CCspDateTime ccsp_datetime_from_milliseconds( int64_t milliseconds );

/* Returns CCSP_DATETIME_NONE if the fields do not describe a real instant */
CSP_C_API_EXPORT CCspDateTime ccsp_datetime_from_parts(
    int year, int month, int day,
    int hour, int minute, int second,
    int nanosecond
);

/* DateTime extraction */
CSP_C_API_EXPORT int64_t ccsp_datetime_to_nanoseconds( CCspDateTime dt );
CSP_C_API_EXPORT int64_t ccsp_datetime_to_seconds( CCspDateTime dt );
CSP_C_API_EXPORT int64_t ccsp_datetime_to_milliseconds( CCspDateTime dt );

/* Any out pointer may be NULL. No-op for CCSP_DATETIME_NONE. */
CSP_C_API_EXPORT void ccsp_datetime_to_parts(
    CCspDateTime dt,
    int* out_year, int* out_month, int* out_day,
    int* out_hour, int* out_minute, int* out_second,
    int* out_nanosecond
);

/* DateTime arithmetic. Saturates rather than overflowing. */
CSP_C_API_EXPORT CCspDateTime ccsp_datetime_add( CCspDateTime dt, CCspTimeDelta delta );
CSP_C_API_EXPORT CCspTimeDelta ccsp_datetime_diff( CCspDateTime a, CCspDateTime b );

/* TimeDelta construction */
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_nanoseconds( int64_t nanoseconds );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_microseconds( int64_t microseconds );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_milliseconds( int64_t milliseconds );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_seconds( double seconds );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_minutes( double minutes );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_hours( double hours );
CSP_C_API_EXPORT CCspTimeDelta ccsp_timedelta_from_days( double days );

/* TimeDelta extraction */
CSP_C_API_EXPORT double ccsp_timedelta_to_seconds( CCspTimeDelta td );
CSP_C_API_EXPORT int64_t ccsp_timedelta_to_nanoseconds( CCspTimeDelta td );

/* Date construction. Returns CCSP_DATE_NONE if the fields are not a real date. */
CSP_C_API_EXPORT CCspDate ccsp_date_from_days( int32_t days_since_epoch );
CSP_C_API_EXPORT CCspDate ccsp_date_from_parts( int year, int month, int day );

/* Date extraction. Any out pointer may be NULL. */
CSP_C_API_EXPORT int32_t ccsp_date_to_days( CCspDate date );
CSP_C_API_EXPORT void ccsp_date_to_parts( CCspDate date, int * out_year, int * out_month, int * out_day );

/* Time (time of day) construction. Returns CCSP_TIME_NONE if out of range. */
CSP_C_API_EXPORT CCspTime ccsp_time_from_nanoseconds( int64_t nanoseconds_since_midnight );
CSP_C_API_EXPORT CCspTime ccsp_time_from_parts( int hour, int minute, int second, int nanosecond );

/* Time extraction. Any out pointer may be NULL. */
CSP_C_API_EXPORT int64_t ccsp_time_to_nanoseconds( CCspTime time );
CSP_C_API_EXPORT void ccsp_time_to_parts( CCspTime time, int * out_hour, int * out_minute, int * out_second, int * out_nanosecond );

#ifdef __cplusplus
}
#endif

#endif /* _IN_CSP_ENGINE_C_CSPTIME_H */
